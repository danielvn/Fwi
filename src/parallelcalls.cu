#include "parallelcalls.h"

__global__ void pre_variables (float *vel, float *epsilon, float *delta, float *v_hor, float *v_po, float *v_nmo, int nx, int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(ix < nx && iz < nz)
    {
        v_po[I(ix,iz)] = vel[I(ix,iz)]*vel[I(ix,iz)];
        v_hor[I(ix,iz)] = v_po[I(ix,iz)]*(1+2*epsilon[I(ix,iz)]);
        v_nmo[I(ix,iz)] = v_po[I(ix,iz)]*(1+2*delta[I(ix,iz)]);
        // v_hor[I(ix,iz)] = 0;
        // v_nmo[I(ix,iz)] = 0;
    }
}

__global__ void CPML_x(float *a_x, float *b_x, int CPML, float Vmax, int nx, float dt, float dh, float f, float d0, float L)
{

    int ix = threadIdx.x + blockDim.x * blockIdx.x;

    if (ix < CPML)
    {
        b_x[ix] = exp(-(d0*Vmax*pow(((CPML-ix)*dh/L),2) + PI*f*(L-(CPML-ix)*dh)/L)*dt);
        a_x[ix] = d0*Vmax*pow((CPML-ix)*dh/L,2)*(b_x[ix]-1)/((d0*Vmax*pow((CPML-ix)*dh/L,2) + PI*f*(L-(CPML-ix)*dh)/L));
    }
    
    if (ix > (nx-CPML-1) && ix < nx)
    {
        b_x[ix] = exp(-(d0*Vmax*pow((ix-nx+CPML+1)*dh/L,2) + PI*f*(L-(ix-nx+CPML+1)*dh)/L)*dt);
        a_x[ix] = d0*Vmax*pow((ix-nx+CPML+1)*dh/L,2)*(b_x[ix]-1)/((d0*Vmax*pow((ix-nx+CPML+1)*dh/L,2) + PI*f*(L-(ix-nx+CPML+1)*dh)/L));
    }
}

__global__ void CPML_z(float *a_z, float *b_z, int CPML, float Vmax, int nz, float dt, float dh, float f, float d0, float L)
{

    int iz = threadIdx.x + blockDim.x * blockIdx.x;

    if (iz > (nz-CPML-1) && iz < nz)
    {
        b_z[iz] = exp(-(d0*Vmax*pow((iz-nz+CPML+1)*dh/L,2) + PI*f*(L-(iz-nz+CPML+1)*dh)/L)*dt);
        a_z[iz] = d0*Vmax*pow((iz-nz+CPML+1)*dh/L,2)*(b_z[iz]-1)/((d0*Vmax*pow((iz-nz+CPML+1)*dh/L,2) + PI*f*(L-(iz-nz+CPML+1)*dh)/L));
    }
}

__global__ void PSI_F (float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, int nx, int nz, float dh, int CPML)
{

    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    float temp_px = 0, temp_fz = 0;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4 && (ix < CPML || ix > (nx-CPML-1) || iz > (nz-CPML-1)))
    {
        temp_px = ((4/5.)*(P[I(ix+1,iz)] - P[I(ix-1,iz)]) - (1/5.)*(P[I(ix+2,iz)] - P[I(ix-2,iz)]) + (4/105.)*(P[I(ix+3,iz)] - P[I(ix-3,iz)]) - (1/280.)*(P[I(ix+4,iz)] - P[I(ix-4,iz)]))/dh;
        temp_fz = ((4/5.)*(F[I(ix,iz+1)] - F[I(ix,iz-1)]) - (1/5.)*(F[I(ix,iz+2)] - F[I(ix,iz-2)]) + (4/105.)*(F[I(ix,iz+3)] - F[I(ix,iz-3)]) - (1/280.)*(F[I(ix,iz+4)] - F[I(ix,iz-4)]))/dh;
        
        psi_px[I(ix,iz)] = b_x[ix]*psi_px[I(ix,iz)] + a_x[ix]*temp_px;
        psi_fz[I(ix,iz)] = b_z[iz]*psi_fz[I(ix,iz)] + a_z[iz]*temp_fz;
    }
}

__global__ void ZETA_F (float *p2, float *r2,float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float *z_px, float *z_fz, float *aten_p, float *aten_f, int nx, int nz, float dh, float dh2, int CPML)
{

    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    float temp_px = 0, temp_fz = 0, temp_x = 0, temp_z = 0;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4 && (ix < CPML || ix > (nx-CPML-1) || iz > (nz-CPML-1)))
    {
        temp_x = ((4/5.)*(psi_px[I(ix+1,iz)] - psi_px[I(ix-1,iz)]) - (1/5.)*(psi_px[I(ix+2,iz)] - psi_px[I(ix-2,iz)]) + (4/105.)*(psi_px[I(ix+3,iz)] - psi_px[I(ix-3,iz)]) - (1/280.)*(psi_px[I(ix+4,iz)] - psi_px[I(ix-4,iz)]))/dh;
        temp_z = ((4/5.)*(psi_fz[I(ix,iz+1)] - psi_fz[I(ix,iz-1)]) - (1/5.)*(psi_fz[I(ix,iz+2)] - psi_fz[I(ix,iz-2)]) + (4/105.)*(psi_fz[I(ix,iz+3)] - psi_fz[I(ix,iz-3)]) - (1/280.)*(psi_fz[I(ix,iz+4)] - psi_fz[I(ix,iz-4)]))/dh;
        
        temp_px = p2[I(ix,iz)] + temp_x;
        temp_fz = r2[I(ix,iz)] + temp_z;
        
        z_px[I(ix,iz)] = b_x[ix]*z_px[I(ix,iz)] + a_x[ix]*temp_px;
        z_fz[I(ix,iz)] = b_z[iz]*z_fz[I(ix,iz)] + a_z[iz]*temp_fz;
        
        aten_p[I(ix,iz)] = temp_x + z_px[I(ix,iz)];
        aten_f[I(ix,iz)] = temp_z + z_fz[I(ix,iz)];
    }
}

__global__ void propagator_F (float *p2, float *r2,float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *v_hor, 
    float *v_po, float *v_nmo, float *aten_p, float *aten_f, float *source, float *shot, float C, float dh2, 
    int nx, int nz, int sx, int sz, int it, int CPML)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;

    float temp_p = 0, temp_f = 0;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4)
    {
        temp_p = dh2*(p2[I(ix,iz)] + aten_p[I(ix,iz)]);
        temp_f = dh2*(r2[I(ix,iz)] + aten_f[I(ix,iz)]);
        
        P_past[I(ix,iz)] = 2*P_pre[I(ix,iz)] - P_past[I(ix,iz)] + v_hor[I(ix,iz)]*C*temp_p + v_po[I(ix,iz)]*C*temp_f;
        F_past[I(ix,iz)] = 2*F_pre[I(ix,iz)] - F_past[I(ix,iz)] + v_nmo[I(ix,iz)]*C*temp_p + v_po[I(ix,iz)]*C*temp_f;
        
        PF[I(ix,iz)] = P_past[I(ix,iz)] + F_past[I(ix,iz)];
    }

    if (ix == (sx-1) && iz == (sz-1)) 
    {
        P_past[I(ix,iz)] += source[it];
        F_past[I(ix,iz)] += source[it];
    }
    
    if (ix > (CPML-1) && ix < (nx-CPML) && iz == (sz-1))
    {
        shot[I(ix,it)] = P_past[I(ix,iz)] + F_past[I(ix,iz)];
    }
}


__global__ void SecondDerivate_P (float *P_pre, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4)
    {
        p2[I(ix,iz)] = (-(205/72.)*P_pre[I(ix,iz)] + (8/5.)*(P_pre[I(ix+1,iz)] + P_pre[I(ix-1,iz)]) - (1/5.)*(P_pre[I(ix+2,iz)] + P_pre[I(ix-2,iz)]) + (8/315.)*(P_pre[I(ix+3,iz)] + P_pre[I(ix-3,iz)]) - (1/560.)*(P_pre[I(ix+4,iz)] + P_pre[I(ix-4,iz)]))/dh2;
        r2[I(ix,iz)] = (-(205/72.)*F_pre[I(ix,iz)] + (8/5.)*(F_pre[I(ix,iz+1)] + F_pre[I(ix,iz-1)]) - (1/5.)*(F_pre[I(ix,iz+2)] + F_pre[I(ix,iz-2)]) + (8/315.)*(F_pre[I(ix,iz+3)] + F_pre[I(ix,iz-3)]) - (1/560.)*(F_pre[I(ix,iz+4)] + F_pre[I(ix,iz-4)]))/dh2;
    }
}


__global__ void res(float* Res, float *shot1, float *shot2, int nx, int nt){
    
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;

    if(ix>=0 && ix<nx && iz>=0 && iz<nt){
        Res[I(ix,iz)] = shot2[I(ix,iz)] - shot1[I(ix,iz)]; 

    }
}

__global__ void retroPropagation(float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *res, float C, float dh2, 
    int nx, int nz, int nt, int sz, int it, int CPML, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4){

        P_past[I(ix,iz)] = 2*P_pre[I(ix,iz)] - P_past[I(ix,iz)] + dh2*C*(p2cx[I(ix,iz)] + atenxcx[I(ix,iz)] + r2dx[I(ix,iz)] + atenzdx[I(ix,iz)]);
        F_past[I(ix,iz)] = 2*F_pre[I(ix,iz)] - F_past[I(ix,iz)] + dh2*C*(p2cz[I(ix,iz)] + r2cz[I(ix,iz)] + atenxcz[I(ix,iz)] + atenzcz[I(ix,iz)]);

        // P_past[I(ix,iz)] = 2*P_pre[I(ix,iz)] - P_past[I(ix,iz)] + dh2*C*(p2cx[I(ix,iz)] + r2dx[I(ix,iz)]);
        // F_past[I(ix,iz)] = 2*F_pre[I(ix,iz)] - F_past[I(ix,iz)] + dh2*C*(p2cz[I(ix,iz)] + r2cz[I(ix,iz)]);

    }

    if (ix > (CPML-1) && ix < (nx-CPML) && iz == (sz-1)) 
    {
        P_past[I(ix,iz)] += res[I(ix,nt-it-1)];
        F_past[I(ix,iz)] += res[I(ix,nt-it-1)];
    }
}

__global__ void entryPoint(float *P_pre, float *F_pre, float *aux1, float *aux2, float *aux3, float *aux4, float *v_hor, float *v_po, float *v_nmo, int nx, int nz){

    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < nx && iz < nz){
        aux1[I(ix,iz)] = v_hor[I(ix,iz)]*P_pre[I(ix,iz)];
        aux2[I(ix,iz)] = v_po[I(ix,iz)]*P_pre[I(ix,iz)];
        aux3[I(ix,iz)] = v_nmo[I(ix,iz)]*F_pre[I(ix,iz)];
        aux4[I(ix,iz)] = v_po[I(ix,iz)]*F_pre[I(ix,iz)];
    }
}

__global__ void PSI_R (float *aux1, float *aux2, float *aux3, float *aux4, float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, int nx, int nz, float dh, int CPML)
{

    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    float p1cx = 0, p1cz = 0, r1dx = 0, r1cz = 0;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4 && (ix < CPML || ix > (nx-CPML-1) || iz > (nz-CPML-1))){
        
        p1cx = ((4/5.)*(aux1[I(ix+1,iz)] - aux1[I(ix-1,iz)]) - (1/5.)*(aux1[I(ix+2,iz)] - aux1[I(ix-2,iz)]) + (4/105.)*(aux1[I(ix+3,iz)] - aux1[I(ix-3,iz)]) - (1/280.)*(aux1[I(ix+4,iz)] - aux1[I(ix-4,iz)]))/dh;
        p1cz = ((4/5.)*(aux2[I(ix,iz+1)] - aux2[I(ix,iz-1)]) - (1/5.)*(aux2[I(ix,iz+2)] - aux2[I(ix,iz-2)]) + (4/105.)*(aux2[I(ix,iz+3)] - aux2[I(ix,iz-3)]) - (1/280.)*(aux2[I(ix,iz+4)] - aux2[I(ix,iz-4)]))/dh;
        r1dx = ((4/5.)*(aux3[I(ix+1,iz)] - aux3[I(ix-1,iz)]) - (1/5.)*(aux3[I(ix+2,iz)] - aux3[I(ix-2,iz)]) + (4/105.)*(aux3[I(ix+3,iz)] - aux3[I(ix-3,iz)]) - (1/280.)*(aux3[I(ix+4,iz)] - aux3[I(ix-4,iz)]))/dh;
        r1cz = ((4/5.)*(aux4[I(ix,iz+1)] - aux4[I(ix,iz-1)]) - (1/5.)*(aux4[I(ix,iz+2)] - aux4[I(ix,iz-2)]) + (4/105.)*(aux4[I(ix,iz+3)] - aux4[I(ix,iz-3)]) - (1/280.)*(aux4[I(ix,iz+4)] - aux4[I(ix,iz-4)]))/dh;

        psixcx[I(ix,iz)] = b_x[ix]*psixcx[I(ix,iz)] + a_x[ix]*p1cx;
        psixcz[I(ix,iz)] = b_z[iz]*psixcz[I(ix,iz)] + a_z[iz]*p1cz;
        psizdx[I(ix,iz)] = b_x[ix]*psizdx[I(ix,iz)] + a_x[ix]*r1dx;
        psizcz[I(ix,iz)] = b_z[iz]*psizcz[I(ix,iz)] + a_z[iz]*r1cz;
    }
}

__global__ void secondDerivate(float *aux1, float *aux2, float *aux3, float *aux4, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float dh2, int nx, int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4){

        
        p2cx[I(ix,iz)] = (-(205/72.)*aux1[I(ix,iz)] + (8/5.)*(aux1[I(ix+1,iz)] + aux1[I(ix-1,iz)]) - (1/5.)*(aux1[I(ix+2,iz)] + aux1[I(ix-2,iz)]) + (8/315.)*(aux1[I(ix+3,iz)] + aux1[I(ix-3,iz)]) - (1/560.)*(aux1[I(ix+4,iz)] + aux1[I(ix-4,iz)]))/dh2;
        p2cz[I(ix,iz)] = (-(205/72.)*aux2[I(ix,iz)] + (8/5.)*(aux2[I(ix,iz+1)] + aux2[I(ix,iz-1)]) - (1/5.)*(aux2[I(ix,iz+2)] + aux2[I(ix,iz-2)]) + (8/315.)*(aux2[I(ix,iz+3)] + aux2[I(ix,iz-3)]) - (1/560.)*(aux2[I(ix,iz+4)] + aux2[I(ix,iz-4)]))/dh2;
        r2dx[I(ix,iz)] = (-(205/72.)*aux3[I(ix,iz)] + (8/5.)*(aux3[I(ix+1,iz)] + aux3[I(ix-1,iz)]) - (1/5.)*(aux3[I(ix+2,iz)] + aux3[I(ix-2,iz)]) + (8/315.)*(aux3[I(ix+3,iz)] + aux3[I(ix-3,iz)]) - (1/560.)*(aux3[I(ix+4,iz)] + aux3[I(ix-4,iz)]))/dh2;
        r2cz[I(ix,iz)] = (-(205/72.)*aux4[I(ix,iz)] + (8/5.)*(aux4[I(ix,iz+1)] + aux4[I(ix,iz-1)]) - (1/5.)*(aux4[I(ix,iz+2)] + aux4[I(ix,iz-2)]) + (8/315.)*(aux4[I(ix,iz+3)] + aux4[I(ix,iz-3)]) - (1/560.)*(aux4[I(ix,iz+4)] + aux4[I(ix,iz-4)]))/dh2;

    }

}

__global__ void ZETA_R (float *p2cx, float *p2cz, float *r2dx, float *r2cz, float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, float *zitaxcx, float *zitaxcz, float *zitazdx, float *zitazcz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz, int nx, int nz, float dh, float dh2, int CPML)
{

    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    float dpsixcx = 0, dpsixcz = 0, dpsizdx = 0, dpsizcz = 0;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4 && (ix < CPML || ix > (nx-CPML-1) || iz > (nz-CPML-1)))
    {
        dpsixcx = ((4/5.)*(psixcx[I(ix+1,iz)] - psixcx[I(ix-1,iz)]) - (1/5.)*(psixcx[I(ix+2,iz)] - psixcx[I(ix-2,iz)]) + (4/105.)*(psixcx[I(ix+3,iz)] - psixcx[I(ix-3,iz)]) - (1/280.)*(psixcx[I(ix+4,iz)] - psixcx[I(ix-4,iz)]))/dh;
        dpsixcz = ((4/5.)*(psixcz[I(ix,iz+1)] - psixcz[I(ix,iz-1)]) - (1/5.)*(psixcz[I(ix,iz+2)] - psixcz[I(ix,iz-2)]) + (4/105.)*(psixcz[I(ix,iz+3)] - psixcz[I(ix,iz-3)]) - (1/280.)*(psixcz[I(ix,iz+4)] - psixcz[I(ix,iz-4)]))/dh;
        dpsizdx = ((4/5.)*(psizdx[I(ix+1,iz)] - psizdx[I(ix-1,iz)]) - (1/5.)*(psizdx[I(ix+2,iz)] - psizdx[I(ix-2,iz)]) + (4/105.)*(psizdx[I(ix+3,iz)] - psizdx[I(ix-3,iz)]) - (1/280.)*(psizdx[I(ix+4,iz)] - psizdx[I(ix-4,iz)]))/dh;
        dpsizcz = ((4/5.)*(psizcz[I(ix,iz+1)] - psizcz[I(ix,iz-1)]) - (1/5.)*(psizcz[I(ix,iz+2)] - psizcz[I(ix,iz-2)]) + (4/105.)*(psizcz[I(ix,iz+3)] - psizcz[I(ix,iz-3)]) - (1/280.)*(psizcz[I(ix,iz+4)] - psizcz[I(ix,iz-4)]))/dh;


        zitaxcx[I(ix,iz)] = b_x[ix]*zitaxcx[I(ix,iz)] + a_x[ix]*(dpsixcx + p2cx[I(ix,iz)]);
        zitaxcz[I(ix,iz)] = b_z[iz]*zitaxcz[I(ix,iz)] + a_z[iz]*(dpsixcz + p2cz[I(ix,iz)]);
        zitazdx[I(ix,iz)] = b_x[ix]*zitazdx[I(ix,iz)] + a_x[ix]*(dpsizdx + r2dx[I(ix,iz)]);
        zitazcz[I(ix,iz)] = b_z[iz]*zitazcz[I(ix,iz)] + a_z[iz]*(dpsizcz + r2cz[I(ix,iz)]);

        atenxcx[I(ix,iz)] = zitaxcx[I(ix,iz)] + dpsixcx;
        atenxcz[I(ix,iz)] = zitaxcz[I(ix,iz)] + dpsixcz;
        atenzdx[I(ix,iz)] = zitazdx[I(ix,iz)] + dpsizdx;
        atenzcz[I(ix,iz)] = zitazcz[I(ix,iz)] + dpsizcz;

        
    }
}

__global__ void GRADIENT(float *P_pre, float *F_pre, float *P_past, float *F_past, float *gcx, float *gcz, float *gdx, float dh2, float dt, int nx, int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    float p2 = 0, r2 = 0;
    if (ix > 3 && ix < nx-4 && iz > 3 && iz < nz-4){

        p2 = (-(205/72.)*P_pre[I(ix,iz)] + (8/5.)*(P_pre[I(ix+1,iz)] + P_pre[I(ix-1,iz)]) - (1/5.)*(P_pre[I(ix+2,iz)] + P_pre[I(ix-2,iz)]) + (8/315.)*(P_pre[I(ix+3,iz)] + P_pre[I(ix-3,iz)]) - (1/560.)*(P_pre[I(ix+4,iz)] + P_pre[I(ix-4,iz)]))/dh2;
        r2 = (-(205/72.)*F_pre[I(ix,iz)] + (8/5.)*(F_pre[I(ix,iz+1)] + F_pre[I(ix,iz-1)]) - (1/5.)*(F_pre[I(ix,iz+2)] + F_pre[I(ix,iz-2)]) + (8/315.)*(F_pre[I(ix,iz+3)] + F_pre[I(ix,iz-3)]) - (1/560.)*(F_pre[I(ix,iz+4)] + F_pre[I(ix,iz-4)]))/dh2;

        gcx[I(ix,iz)] += dt*p2*P_past[I(ix,iz)];
        gcz[I(ix,iz)] += dt*r2*(P_past[I(ix,iz)] + F_past[I(ix,iz)]);
        gdx[I(ix,iz)] += dt*p2*F_past[I(ix,iz)];
    }
}

__global__ void NORMAL(float *gcx, float *gcz, float *gdx, int nx, int nz, float ngcx, float ngcz, float ngdx)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx  && iz < nz && iz > layer){
        gcx[I(ix,iz)] = gcx[I(ix,iz)]/ngcx;
        gcz[I(ix,iz)] = gcz[I(ix,iz)]/ngcz;
        gdx[I(ix,iz)] = gdx[I(ix,iz)]/ngdx;
    }
}

__global__ void UpdateModel(float *gcx, float *gcz, float *gdx, float *v_hor, float *v_po, float *v_nmo, float alpha1,  float alpha2, float alpha3, int nx, int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx && iz < nz && iz > layer){
        v_po[I(ix,iz)] = v_po[I(ix,iz)] - (alpha2*gcz[I(ix,iz)]);
        v_nmo[I(ix,iz)] = v_nmo[I(ix,iz)] - (alpha3*gdx[I(ix,iz)]);
        v_hor[I(ix,iz)] = v_hor[I(ix,iz)] - (alpha1*gcx[I(ix,iz)]);
        if(v_hor[I(ix,iz)]<v_nmo[I(ix,iz)])
            v_hor[I(ix,iz)]=v_nmo[I(ix,iz)];
        
    }
}

__global__ void compute_sy (float *s_vp, float *s_vn, float *s_vh, float *y_b1, float *y_b2, float *y_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, float *b1, float *b2, float *b3, float *back_b1, float *back_b2, float *back_b3, int nx, int nz, int k)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx && iz< nz)
    {
        s_vp[I(ix,iz)+k*nx*nz] = vp[I(ix,iz)] - back_vp[I(ix,iz)];
        s_vn[I(ix,iz)+k*nx*nz] = vn[I(ix,iz)] - back_vn[I(ix,iz)];
        s_vh[I(ix,iz)+k*nx*nz] = vh[I(ix,iz)] - back_vh[I(ix,iz)];
        y_b1[I(ix,iz)+k*nx*nz] = b1[I(ix,iz)] - back_b1[I(ix,iz)];
        y_b2[I(ix,iz)+k*nx*nz] = b2[I(ix,iz)] - back_b2[I(ix,iz)];
        y_b3[I(ix,iz)+k*nx*nz] = b3[I(ix,iz)] - back_b3[I(ix,iz)];
    }
}

__global__ void compute_qi (float *q_b1, float *q_b2, float *q_b3, float *y_b1, float *y_b2, float *y_b3, float epsilon_v, float epsilon_d, float epsilon_e, int nx, int nz, int k)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx && iz < nz)
    {
        q_b1[I(ix,iz)] -= epsilon_v*y_b1[I(ix,iz)+k*nx*nz];
        q_b2[I(ix,iz)] -= epsilon_d*y_b2[I(ix,iz)+k*nx*nz];
        q_b3[I(ix,iz)] -= epsilon_e*y_b3[I(ix,iz)+k*nx*nz];
    }
}

__global__ void compute_ri (float *r_b1, float *r_b2, float *r_b3, float *q_b1, float *q_b2, float *q_b3, float gamma_v, float gamma_d, float gamma_e, int nx, int nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx && iz < nz)
    {
        r_b1[I(ix,iz)] = gamma_v*q_b1[I(ix,iz)];
        r_b2[I(ix,iz)] = gamma_d*q_b2[I(ix,iz)];
        r_b3[I(ix,iz)] = gamma_e*q_b3[I(ix,iz)];
    }
}

__global__ void update_r (float *r_b1, float *r_b2, float *r_b3, float *s_vp, float *s_vn, float *s_vh, float epsilon_v, float epsilon_d, float epsilon_e, float beta_v, float beta_d, float beta_e, int nx, int nz, int k)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx && iz < nz)
    {
        r_b1[I(ix,iz)] += (epsilon_v - beta_v)*s_vp[I(ix,iz)+k*nx*nz];
        r_b2[I(ix,iz)] += (epsilon_d - beta_d)*s_vn[I(ix,iz)+k*nx*nz];
        r_b3[I(ix,iz)] += (epsilon_e - beta_e)*s_vh[I(ix,iz)+k*nx*nz];
    }
}

__global__ void update_LBFGS (float *r_b1, float *r_b2, float *r_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, int nx, int nz, float alpha_v, float alpha_d, float alpha_e, int flag_v, int flag_d, int flag_e, int flag_min, int flag_max)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iz = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (ix < nx && iz > layer && iz < nz)
    {
        if (flag_v == 1)
        {
        vp[I(ix,iz)] = back_vp[I(ix,iz)] - alpha_v*r_b1[I(ix,iz)];
        
        if (flag_max == 1)
        {
            if (vp[I(ix,iz)] > max_v)
              vp[I(ix,iz)] = max_v;
        }
        
        if (flag_min == 1)
        {
            if (vp[I(ix,iz)] < min_v)
              vp[I(ix,iz)] = min_v;
        }
        }
        
        if (flag_d == 1)
        {
        vn[I(ix,iz)] = back_vn[I(ix,iz)] - alpha_d*r_b2[I(ix,iz)];
        
        if (flag_max == 1)
        {
            if (vn[I(ix,iz)] > max_d)
              vn[I(ix,iz)] = max_d;
        }
        
        if (flag_min == 1)
        {
            if (vn[I(ix,iz)] < min_d)
              vn[I(ix,iz)] = min_d;
        }
        } 
        
        if (flag_e == 1)
        {
        vh[I(ix,iz)] = back_vh[I(ix,iz)] - alpha_e*r_b3[I(ix,iz)];
        
        if (flag_max == 1)
        {
            if (vh[I(ix,iz)] > max_e)
              vh[I(ix,iz)] = max_e;
        }
        
        if (flag_min == 1)
        {
            if (vh[I(ix,iz)] < min_e)
              vh[I(ix,iz)] = min_e;
        }
        
        if (vh[I(ix,iz)] < vn[I(ix,iz)])
          vh[I(ix,iz)] = vn[I(ix,iz)];
        }  
    }
}
