# include "serialcalls.h"
# include "parallelcalls.h"

void CPML_x_h(float *a_x, float *b_x, int CPML, float Vmax, int nx, float dt, float dh, float f, float d0, float L){
    dim3 Grid_CPML_x(ceil(nx/(float)TILE_WIDTH_X));
    dim3 Block_CPML(TILE_WIDTH_X);
    CPML_x <<<Grid_CPML_x,Block_CPML>>>(a_x, b_x, CPML, Vmax, nx, dt, dh, f, d0, L);
}


void CPML_z_h(float *a_z, float *b_z, int CPML, float Vmax, int nz, float dt, float dh, float f, float d0, float L){
    dim3 Grid_CPML_z(ceil(nz/(float)TILE_WIDTH_X));
    dim3 Block_CPML(TILE_WIDTH_X);
    CPML_z <<<Grid_CPML_z,Block_CPML>>>(a_z, b_z, CPML, Vmax, nz, dt, dh, f, d0, L);
}

void pre_variables_h(float *vel, float *epsilon, float *delta, float *v_hor, float *v_po, float *v_nmo, int nx, int nz){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    pre_variables <<<Grid,Block>>>(vel, epsilon, delta, v_hor, v_po, v_nmo, nx, nz);
}

void PSI_F_h(float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, int nx, int nz, float dh, int CPML){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    PSI_F <<<Grid,Block>>>(P, F, a_x, b_x, a_z, b_z, psi_px, psi_fz, nx, nz, dh, CPML);

}

void SecondDerivate_P_h(float *P_pre, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    SecondDerivate_P <<<Grid,Block>>>(P_pre, F_pre, p2, r2, dh2, nx, nz);

}

void ZETA_F_h(float *p2, float *r2, float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float *z_px, float *z_fz, float *aten_p, float *aten_f, int nx, int nz, float dh, float dh2, int CPML){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    ZETA_F <<<Grid,Block>>>(p2, r2, P, F, a_x, b_x, a_z, b_z, psi_px, psi_fz, z_px, z_fz, aten_p, aten_f, nx, nz, dh, dh2, CPML);
}

void propagator_F_h(float *p2, float *r2, float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *v_hor, float *v_po, float *v_nmo, float *aten_p, float *aten_f, float *source, float *shot, float C, float dh2, int nx, int nz, int sx, int sz, int it, int CPML){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    propagator_F <<<Grid,Block>>>(p2, r2, PF, P_pre, F_pre, P_past, F_past, v_hor, v_po, v_nmo, aten_p, aten_f, source, shot, C, dh2, nx, nz, sx, sz, it, CPML);
}

void res_h(float* Res, float *shot1, float *shot2, int nx, int nt){
    dim3 Grid2(ceil(nx/(float)32),ceil(nt/(float)32));
    dim3 Block2(32,32);
    res<<<Grid2,Block2>>>(Res, shot1, shot2, nx, nt);
}

void launchPropagation(cudaStream_t stream1, cudaStream_t stream2, float *PF_h, float *PF, float *P_past, float *P_pre, float *F_past, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz,
	float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float dh, int CPML, float *z_px, float *z_fz, float *aten_p, float *aten_f, float *v_hor, float *v_po,
	float *v_nmo, float *source, float *shot2, float C, int sx, int sz, int it){
	dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
	cudaMemcpyAsync(PF_h, P_past, 2*nx*nz*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    SecondDerivate_P <<<Grid,Block, 0, stream2>>>(P_pre, F_pre, p2, r2, dh2, nx, nz);
    PSI_F <<<Grid,Block, 0, stream2>>>(P_pre, F_pre, a_x, b_x, a_z, b_z, psi_px, psi_fz, nx, nz, dh, CPML);
    ZETA_F <<<Grid,Block, 0, stream2>>>(p2, r2, P_pre, F_pre, a_x, b_x, a_z, b_z, psi_px, psi_fz, z_px, z_fz, aten_p, aten_f, nx, nz, dh, dh2, CPML);
    propagator_F <<<Grid,Block>>>(p2, r2, PF, P_pre, F_pre, P_past, F_past, v_hor, v_po, v_nmo, aten_p, aten_f, source, shot2, 
    C, dh2, nx, nz, sx, sz, it, CPML);
}

void launchBackPropagation(cudaStream_t stream1, cudaStream_t stream2, int nx, int nz, int nt, float *PF_h, float *P_pre, float *P_past, float *F_pre, float *F_past, float *aux1,
	float *aux2, float *aux3, float *aux4, float *v_hor, float *v_po, float *v_nmo, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float dh2, float *P_pre2, float *F_pre2,
	float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, float dh, int CPML, float *zitaxcx, float *zitaxcz, float *zitazdx,
	float *zitazcz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz, float *Res, float C, float *PF, int it, int sx, int sz, float dt, float *gcx, float *gcz, float *gdx)
{
	dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);

    cudaMemcpyAsync(P_pre2, PF_h, 2*nx*nz*sizeof(float), cudaMemcpyHostToDevice, stream1);
    entryPoint<<<Grid,Block, 0, stream2>>>(P_pre, F_pre, aux1, aux2, aux3, aux4, v_hor, v_po, v_nmo, nx, nz);
        
    secondDerivate<<<Grid,Block, 0, stream2>>>(aux1, aux2, aux3, aux4, p2cx, p2cz, r2dx, r2cz, dh2, nx, nz);

    PSI_R<<<Grid,Block, 0, stream2>>>(aux1, aux2, aux3, aux4, a_x, b_x, a_z, b_z, psixcx, psixcz, psizdx, psizcz, nx, nz, dh, CPML);

    ZETA_R<<<Grid,Block, 0, stream2>>>(p2cx, p2cz, r2dx, r2cz, a_x, b_x, a_z, b_z, psixcx, psixcz, psizdx,
    psizcz, zitaxcx, zitaxcz, zitazdx, zitazcz, atenxcx, atenxcz, atenzdx, atenzcz, nx, nz, dh, dh2, CPML);
        
    retroPropagation<<<Grid,Block, 0, stream2>>>(PF, P_pre, F_pre, P_past, F_past, Res, C, dh2, 
    nx, nz, nt, sz, it, CPML, p2cx, p2cz, r2dx, r2cz, atenxcx, atenxcz, atenzdx, atenzcz);
    GRADIENT<<<Grid,Block>>>(P_pre2, F_pre2, P_past, F_past, gcx, gcz, gdx, dh2, dt, nx, nz); 

}

void NORMAL_h(float *gcx, float *gcz, float *gdx, int nx, int nz, float ngcx, float ngcz, float ngdx){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    NORMAL<<<Grid,Block>>>(gcx, gcz, gdx, nx, nz, ngcx, ngcz, ngdx);
}

void UpdateModel_h(float *gcx, float *gcz, float *gdx, float *v_hor, float *v_po, float *v_nmo, float alpha1,  float alpha2, float alpha3, int nx, int nz){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    UpdateModel<<<Grid,Block>>>(gcx, gcz, gdx, v_hor, v_po, v_nmo, alpha1, alpha2, alpha3, nx, nz);

}

void compute_sy_h (float *s_vp, float *s_vn, float *s_vh, float *y_b1, float *y_b2, float *y_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, float *b1, float *b2, float *b3, float *back_b1, float *back_b2, float *back_b3, int nx, int nz, int k){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    compute_sy<<<Grid,Block>>>(s_vp, s_vn, s_vh, y_b1, y_b2, y_b3, vp, vn, vh, back_vp, back_vn, back_vh, b1, b2, b3, back_b1, back_b2, back_b3, nx, nz, k);
}

void compute_qi_h (float *q_b1, float *q_b2, float *q_b3, float *y_b1, float *y_b2, float *y_b3, float epsilon_v, float epsilon_d, float epsilon_e, int nx, int nz, int k){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    compute_qi <<<Grid,Block>>>(q_b1, q_b2, q_b3, y_b1, y_b2, y_b3, epsilon_v, epsilon_d, epsilon_e, nx, nz, k);
}

void compute_ri_h (float *r_b1, float *r_b2, float *r_b3, float *q_b1, float *q_b2, float *q_b3, float gamma_v, float gamma_d, float gamma_e, int nx, int nz){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    compute_ri <<<Grid,Block>>> (r_b1, r_b2, r_b3, q_b1, q_b2, q_b3, gamma_v, gamma_d, gamma_e, nx, nz);
}

void update_r_h (float *r_b1, float *r_b2, float *r_b3, float *s_vp, float *s_vn, float *s_vh, float epsilon_v, float epsilon_d, float epsilon_e, float beta_v, float beta_d, float beta_e, int nx, int nz, int k){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    update_r <<<Grid,Block>>> (r_b1, r_b2, r_b3, s_vp, s_vn, s_vh, epsilon_v, epsilon_d, epsilon_e, beta_v, beta_d, beta_e, nx, nz, k);
}

void update_LBFGS_h (float *r_b1, float *r_b2, float *r_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, int nx, int nz, float alpha_v, float alpha_d, float alpha_e, int flag_v, int flag_d, int flag_e, int flag_min, int flag_max){
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    update_LBFGS <<<Grid,Block>>> (r_b1, r_b2, r_b3, vp, vn, vh, back_vp, back_vn, back_vh, nx, nz, alpha_v, alpha_d, alpha_e, flag_v, flag_d, flag_e, flag_min, flag_max);
}