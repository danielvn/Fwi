# include <stdio.h>
# include <stdlib.h>
# include <string.h>  
# include <math.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include <time.h>
# include <mpi.h>

# include "parallelcalls.h"
# include "serialcalls.h"
# define iterations 100
# define it_LBFGS 1
# define mScale 2


int main()
{
    /***************** PARAMETERS MPI *******************/

    int rank, rsize;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&rsize);
    
    if((rank%2)==0){
        cudaSetDevice(0);
    }
    else{
        cudaSetDevice(1);
    }
    /******************* PARAMETERS *********************/

    clock_t begin, end;
    float time_spent;


    int CPML = 20;
    int nx = 841;
    int nz = 270;
    float dh = 20;
    float dt = 2e-3;
    float tend = 5;
    int nt = ceil(tend/dt);
    int sz = 5;
    int sx = CPML;

    float R = 1e-6; 
    float L = CPML*dh;
    float d0 = -3*log(R)/(2*L);
    float Vmax = 6000;
    float f = 3;

    float dh2 = dh*dh;
    float C = (dt*dt)/dh2;
    int nshots = 15;
    int space = (int)floor(((float)(nx-2*CPML-1))/(float)nshots);

    int k = 0, m = 5, sw = 0, count = 0;
    float gamma_v = 0, gamma_d = 0, gamma_e = 0;
    float beta_v = 0, beta_d = 0, beta_e = 0;
    float alpha_v_LBFGS = 1, alpha_d_LBFGS = 1, alpha_e_LBFGS = 1;

    float *source_h = (float*)calloc(nt,sizeof(float));
    float *vel_h = (float*)calloc(nx*nz,sizeof(float));
    float *epsilon_h = (float*)calloc(nx*nz,sizeof(float));
    float *delta_h = (float*)calloc(nx*nz,sizeof(float));
    float *veli_h = (float*)calloc(nx*nz,sizeof(float));
    float *epsiloni_h = (float*)calloc(nx*nz,sizeof(float));
    float *deltai_h = (float*)calloc(nx*nz,sizeof(float));
    // float *PF_h = (float*)calloc(nx*nt*nz, sizeof(float));
    
    float *PF_h;
    cudaMallocHost((void**)&PF_h, 2*nt*nx*nz*sizeof(float));
    

    float *F_h = (float*)calloc(nx*nt*nz, sizeof(float));
    float *residual_h = (float*)calloc(nx*nt, sizeof(float));
    float *gcx_h = (float*)calloc(nx*nz,sizeof(float));
    float *gcz_h = (float*)calloc(nx*nz,sizeof(float));
    float *gdx_h = (float*)calloc(nx*nz,sizeof(float));
    float *norm1_h = (float*)calloc(nx*nz,sizeof(float));
    float *norm2_h = (float*)calloc(nx*nz,sizeof(float));
    float *sigma_v = (float*)calloc(m,sizeof(float));
    float *sigma_d = (float*)calloc(m,sizeof(float));
    float *sigma_e = (float*)calloc(m,sizeof(float));
    float *epsilon_v = (float*)calloc(m,sizeof(float));
    float *epsilon_d = (float*)calloc(m,sizeof(float));
    float *epsilon_e = (float*)calloc(m,sizeof(float));
    float *phi_function = (float*)calloc(iterations,sizeof(float));

    int *posShots = (int*)calloc(nshots+1,sizeof(int));
    float *gcx_h_m = (float*)calloc(nx*nz,sizeof(float));
    float *gcz_h_m = (float*)calloc(nx*nz,sizeof(float));
    float *gdx_h_m = (float*)calloc(nx*nz,sizeof(float));

    
    /*************** LOAD SOURCE AND MODEL ****************/
    int it = 0;
    
    FILE *M = fopen("vp_ori.bin","rb");
    fread(vel_h,nx*nz,sizeof(float),M);
    fclose(M);
    
    FILE *E = fopen("vh_ori.bin","rb");
    fread(epsilon_h,nx*nz,sizeof(float),E);
    fclose(E);
    
    FILE *D = fopen("vn_ori.bin","rb");
    fread(delta_h,nx*nz,sizeof(float),D);
    fclose(D);
    
    M = fopen("vp_ini.bin","rb");
    fread(veli_h,nx*nz,sizeof(float),M);
    fclose(M);

    E = fopen("vh_ini.bin","rb");
    fread(epsiloni_h,nx*nz,sizeof(float),E);
    fclose(E);

    D = fopen("vn_ini.bin","rb");
    fread(deltai_h,nx*nz,sizeof(float),D);
    fclose(D);

    /********************** CUDA ***************************/
    float *source, *vel, *epsilon, *delta, *veli, *epsiloni, *deltai, *v_hor, *v_po, *v_nmo, *P_pre, *P_past, *F_pre, *F_past, *PF, *shot1, *shot2;
    float *a_x, *a_z, *b_x, *b_z, *psi_px, *psi_fz, *z_px, *z_fz, *aten_p, *aten_f, *Res;
    float *p2cx, *p2cz, *r2dx, *r2cz;
    float *aux1, *aux2, *aux3, *aux4;
    float *psixcx, *psixcz, *psizdx, *psizcz;
    float *atenxcx, *atenxcz, *atenzdx, *atenzcz;
    float *zitaxcx, *zitaxcz, *zitazdx, *zitazcz;
    float *gcx, *gcz, *gdx;
    float *p2, *r2;
    float *P_pre2, *F_pre2;
    float *back_gcx, *back_gcz, *back_gdx, *back_vhor, *back_vpo, *back_vnmo;
    float *s_vp, *s_vh, *s_vn, *y_gcz, *y_gcx, *y_gdx;
    float *q_gcz, *q_gcx, *q_gdx, *r_b1, *r_b2, *r_b3;
    
    cudaMalloc((void **) &source, nt*sizeof(float));
    cudaMalloc((void **) &vel, nx*nz*sizeof(float));
    cudaMalloc((void **) &epsilon, nx*nz*sizeof(float));
    cudaMalloc((void **) &delta, nx*nz*sizeof(float));
    cudaMalloc((void **) &veli, nx*nz*sizeof(float));
    cudaMalloc((void **) &epsiloni, nx*nz*sizeof(float));
    cudaMalloc((void **) &deltai, nx*nz*sizeof(float));
    cudaMalloc((void **) &v_hor, nx*nz*sizeof(float));
    cudaMalloc((void **) &v_po, nx*nz*sizeof(float));
    cudaMalloc((void **) &v_nmo, nx*nz*sizeof(float));
    
    cudaMalloc((void **) &a_x, nx*sizeof(float));
    cudaMalloc((void **) &a_z, nz*sizeof(float));
    cudaMalloc((void **) &b_x, nx*sizeof(float));
    cudaMalloc((void **) &b_z, nz*sizeof(float));
    cudaMalloc((void **) &psi_px, nx*nz*sizeof(float));
    cudaMalloc((void **) &psi_fz, nx*nz*sizeof(float));
    cudaMalloc((void **) &z_px, nx*nz*sizeof(float));
    cudaMalloc((void **) &z_fz, nx*nz*sizeof(float));
    cudaMalloc((void **) &aten_p, nx*nz*sizeof(float));
    cudaMalloc((void **) &aten_f, nx*nz*sizeof(float));

    cudaMalloc((void **) &P_pre, 2*nx*nz*sizeof(float));
    cudaMalloc((void **) &P_past, 2*nx*nz*sizeof(float));
    F_pre = &P_pre[nx*nz];
    F_past = &P_past[nx*nz];
    cudaMalloc((void **) &P_pre2, 2*nx*nz*sizeof(float));
    F_pre2=&P_pre2[nx*nz];
    
    cudaMalloc((void **) &PF, nx*nz*sizeof(float));
    cudaMalloc((void **) &shot1, nx*nt*(nshots+1)*sizeof(float));
    cudaMalloc((void **) &shot2, nx*nt*sizeof(float));
    cudaMalloc((void **) &Res, nx*nt*sizeof(float));

    cudaMalloc((void **) &aux1, nx*nz*sizeof(float));
    cudaMalloc((void **) &aux2, nx*nz*sizeof(float));
    cudaMalloc((void **) &aux3, nx*nz*sizeof(float));
    cudaMalloc((void **) &aux4, nx*nz*sizeof(float));

    cudaMalloc((void **) &atenxcx, nx*nz*sizeof(float));
    cudaMalloc((void **) &atenxcz, nx*nz*sizeof(float));
    cudaMalloc((void **) &atenzdx, nx*nz*sizeof(float));
    cudaMalloc((void **) &atenzcz, nx*nz*sizeof(float));

    zitaxcx = z_px;
    zitaxcz = z_fz;
    zitazdx = aten_f;
    zitazcz = aten_p;
    

    cudaMalloc((void **) &psixcx, nx*nz*sizeof(float));
    cudaMalloc((void **) &psixcz, nx*nz*sizeof(float));
    psizdx = psi_px;
    psizcz = psi_fz;
    

    cudaMalloc((void **) &p2cx, nx*nz*sizeof(float));
    cudaMalloc((void **) &p2cz, nx*nz*sizeof(float));
    cudaMalloc((void **) &r2dx, nx*nz*sizeof(float));
    cudaMalloc((void **) &r2cz, nx*nz*sizeof(float));

    cudaMalloc((void **) &gcx, nx*nz*sizeof(float));
    cudaMalloc((void **) &gcz, nx*nz*sizeof(float));
    cudaMalloc((void **) &gdx, nx*nz*sizeof(float));

    cudaMalloc((void **) &p2, nx*nz*sizeof(float));
    cudaMalloc((void **) &r2, nx*nz*sizeof(float));

    cudaMalloc((void **) &back_vpo, nx*nz*sizeof(float));
    cudaMalloc((void **) &back_vnmo, nx*nz*sizeof(float));
    cudaMalloc((void **) &back_vhor, nx*nz*sizeof(float));
    cudaMalloc((void **) &back_gcz, nx*nz*sizeof(float));
    cudaMalloc((void **) &back_gcx, nx*nz*sizeof(float));
    cudaMalloc((void **) &back_gdx, nx*nz*sizeof(float));

    cudaMalloc((void **) &s_vp, m*nx*nz*sizeof(float));
    cudaMalloc((void **) &s_vn, m*nx*nz*sizeof(float));
    cudaMalloc((void **) &s_vh, m*nx*nz*sizeof(float));
    cudaMalloc((void **) &y_gcz, m*nx*nz*sizeof(float));
    cudaMalloc((void **) &y_gcx, m*nx*nz*sizeof(float));
    cudaMalloc((void **) &y_gdx, m*nx*nz*sizeof(float));


    cudaMalloc((void **) &q_gcz, nx*nz*sizeof(float));
    cudaMalloc((void **) &q_gcx, nx*nz*sizeof(float));
    cudaMalloc((void **) &q_gdx, nx*nz*sizeof(float));
    cudaMalloc((void **) &r_b1, nx*nz*sizeof(float));
    cudaMalloc((void **) &r_b2, nx*nz*sizeof(float));
    cudaMalloc((void **) &r_b3, nx*nz*sizeof(float));


    cudaMemcpy(vel, vel_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon, epsilon_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta, delta_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(veli, veli_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(epsiloni, epsiloni_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deltai, deltai_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);

    // free(source_h);
    free(vel_h);
    free(epsilon_h);
    free(delta_h);
    free(veli_h);
    free(epsiloni_h);
    free(deltai_h);
    
    
    // dim3 Grid_CPML_x(ceil(nx/(float)TILE_WIDTH_X));
    // dim3 Grid_CPML_z(ceil(nz/(float)TILE_WIDTH_X));
    // dim3 Block_CPML(TILE_WIDTH_X);
    
    // dim3 Grid(ceil((nx*1)/(float)32),ceil((nz*1)/(float)32));
    // dim3 Block(32,32);

    // dim3 Grid2(ceil(nx/(float)32),ceil(nt/(float)32));
    // dim3 Block2(32,32);

    
    
    /******************* KERNELS *******************/
    posShots[0]= CPML;
    for (int iter = 1; iter <= nshots; ++iter)
        posShots[iter] = posShots[iter-1] + space;



    if(rank==0)
        printf("%s\n", "Iniciando streams...");
    
    cudaStream_t stream1, stream2;
    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    FILE *field;
    for(int scale = 0; scale < mScale; scale++){
        cudaMemset(a_x, 0, nx*sizeof(float));
        cudaMemset(b_x, 0, nx*sizeof(float));
        cudaMemset(a_z, 0, nz*sizeof(float));
        cudaMemset(b_z, 0, nz*sizeof(float));


        cudaMemset(psi_px, 0, nx*nz*sizeof(float));
        cudaMemset(psi_fz, 0, nx*nz*sizeof(float));
        cudaMemset(z_px, 0, nx*nz*sizeof(float));
        cudaMemset(z_fz, 0, nx*nz*sizeof(float));
        cudaMemset(aten_p, 0, nx*nz*sizeof(float));
        cudaMemset(aten_f, 0, nx*nz*sizeof(float));
        cudaMemset(P_pre, 0, nx*nz*sizeof(float));
        cudaMemset(P_past, 0, nx*nz*sizeof(float));
        cudaMemset(F_pre, 0, nx*nz*sizeof(float));
        cudaMemset(F_past, 0, nx*nz*sizeof(float));
        cudaMemset(PF, 0, nx*nz*sizeof(float));
        cudaMemset(shot1, 0, nx*nt*(nshots+1)*sizeof(float));
        cudaMemset(shot2, 0, nx*nt*(nshots+1)*sizeof(float));
        cudaMemset(aux1, 0, nx*nz*sizeof(float));
        cudaMemset(aux2, 0, nx*nz*sizeof(float));
        cudaMemset(aux3, 0, nx*nz*sizeof(float));
        cudaMemset(aux4, 0, nx*nz*sizeof(float));
        cudaMemset(p2cx, 0, nx*nz*sizeof(float));
        cudaMemset(p2cz, 0, nx*nz*sizeof(float));
        cudaMemset(r2dx, 0, nx*nz*sizeof(float));
        cudaMemset(r2cz, 0, nx*nz*sizeof(float));
        cudaMemset(gcx, 0, nx*nz*sizeof(float));
        cudaMemset(gcz, 0, nx*nz*sizeof(float));
        cudaMemset(gdx, 0, nx*nz*sizeof(float));
        sw = 0;
        k = 0;
        alpha_v_LBFGS = 1;
        alpha_d_LBFGS = 1;
        alpha_e_LBFGS = 1;
        count = 0;
        gamma_v = 0;
        gamma_d = 0;
        gamma_e = 0;
        beta_v = 0;
        beta_d = 0;
        beta_e = 0;
        CPML_x_h(a_x, b_x, CPML, Vmax, nx, dt, dh, f, d0, L);
        CPML_z_h(a_z, b_z, CPML, Vmax, nz, dt, dh, f, d0, L);

        pre_variables_h(vel, epsilon, delta, v_hor, v_po, v_nmo, nx, nz);
        
        begin = clock();

        if(rank==0)
            printf("%s\n", "Calculo de Shots...");

        float t = 0;
        for (it=0;it<nt;it++){
            t += dt;
            source_h[it] = 1*(1-2*pow(PI*f*(t-(1/f)),2))*exp(-pow(PI*f*(t-(1/f)),2));
        }
        cudaMemcpy(source, source_h, nt*sizeof(float), cudaMemcpyHostToDevice);
        for(int i = rank; i<=nshots; i+=rsize){

            cudaMemset(psi_px, 0, nx*nz*sizeof(float));
            cudaMemset(psi_fz, 0, nx*nz*sizeof(float));
            cudaMemset(z_px, 0, nx*nz*sizeof(float));
            cudaMemset(z_fz, 0, nx*nz*sizeof(float));
            cudaMemset(aten_p, 0, nx*nz*sizeof(float));
            cudaMemset(aten_f, 0, nx*nz*sizeof(float));
            cudaMemset(P_pre, 0, nx*nz*sizeof(float));
            cudaMemset(P_past, 0, nx*nz*sizeof(float));
            cudaMemset(F_pre, 0, nx*nz*sizeof(float));
            cudaMemset(F_past, 0, nx*nz*sizeof(float)); 
        

            for (it=0;it<nt;it++){


                PSI_F_h(P_pre, F_pre, a_x, b_x, a_z, b_z, psi_px, psi_fz, nx, nz, dh, CPML);
                SecondDerivate_P_h(P_pre, F_pre, p2, r2, dh2, nx, nz);
                ZETA_F_h(p2, r2, P_pre, F_pre, a_x, b_x, a_z, b_z, psi_px, psi_fz, z_px, z_fz, aten_p, aten_f, nx, nz, dh, dh2, CPML);
                propagator_F_h(p2, r2, PF, P_pre, F_pre, P_past, F_past, v_hor, v_po, v_nmo, aten_p, aten_f, source, &shot1[nx*nt*i], 
                C, dh2, nx, nz, posShots[i], sz, it, CPML);


                float *URSS = P_past;
                P_past = P_pre;
                P_pre = URSS;
        
                float *URSS2 = F_past;
                F_past = F_pre;
                F_pre = URSS2;
            }    
        }

        end = clock();
        time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        
        if(rank==0)
            printf("time: %e\n", time_spent);


        // field=fopen("field1.bin", "wb");
        // fwrite(PF_h,sizeof(float),nx*nz*nt, field);
        // fclose(field);


        
        if(scale==0){
            pre_variables_h(veli, epsiloni, deltai, v_hor, v_po, v_nmo, nx, nz);
        }else{
            cudaMemcpy(v_hor, gcx_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(v_po, gcz_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(v_nmo, gdx_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
        }
        // cudaMemcpy(v_po, veli_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(v_hor, epsiloni_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(v_nmo, deltai_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
        
        if(rank==0)
            printf("%s\n", "Inicio de la FWI...");
        
        
        for(int n = 0; n<iterations; n++){
            float fiTotal = 0;
            memset(gcz_h_m, 0, nx*nz*sizeof(float));
            memset(gcx_h_m, 0, nx*nz*sizeof(float));
            memset(gdx_h_m, 0, nx*nz*sizeof(float));
            memset(gcz_h, 0, nx*nz*sizeof(float));
            memset(gcx_h, 0, nx*nz*sizeof(float));
            memset(gdx_h, 0, nx*nz*sizeof(float));
            cudaMemset(gcx, 0, nx*nz*sizeof(float));
            cudaMemset(gcz, 0, nx*nz*sizeof(float));
            cudaMemset(gdx, 0, nx*nz*sizeof(float));
            begin = clock();
            for(int i = rank; i<=nshots; i+=rsize){

                cudaMemset(psi_px, 0, nx*nz*sizeof(float));
                cudaMemset(psi_fz, 0, nx*nz*sizeof(float));
                cudaMemset(aten_p, 0, nx*nz*sizeof(float));
                cudaMemset(aten_f, 0, nx*nz*sizeof(float));
                cudaMemset(z_px, 0, nx*nz*sizeof(float));
                cudaMemset(z_fz, 0, nx*nz*sizeof(float));
                cudaMemset(P_pre, 0, nx*nz*sizeof(float));
                cudaMemset(P_past, 0, nx*nz*sizeof(float));
                cudaMemset(F_pre, 0, nx*nz*sizeof(float));
                cudaMemset(F_past, 0, nx*nz*sizeof(float));


                for (it=0;it<nt;it++){

                    launchPropagation(stream1, stream2, &PF_h[2*it*nx*nz], PF, P_past, P_pre, F_past, F_pre, p2, r2, dh2, nx, nz, a_x, b_x, a_z, b_z, psi_px, psi_fz, dh, CPML, z_px, z_fz,
                     aten_p, aten_f, v_hor, v_po, v_nmo, source, shot2, C, posShots[i], sz, it);

                    float *URSS = P_past;
                    P_past = P_pre;
                    P_pre = URSS;
            
                    float *URSS2 = F_past;
                    F_past = F_pre;
                    F_pre = URSS2;
                }
                
                
                res_h(Res, &shot1[nx*nt*i], shot2, nx, nt);
                cudaMemcpy(residual_h, Res, nx*nt*sizeof(float), cudaMemcpyDeviceToHost);
                // field=fopen("field2.bin", "wb");
                // fwrite(PF_h,sizeof(float),nx*nz*nt, field);
                // fclose(field);


                // field=fopen("res.bin", "wb");
                // fwrite(residual_h,sizeof(float),nx*nt, field);
                // fclose(field);
                
                float fi = 0;
                
                for(int j = 0; j<(nx*nt); j++){
                    fi += powf(residual_h[j],2);
                }
                fi = fi/2;
                fiTotal += fi;
                // printf("%f\n", fi);

                cudaMemset(P_pre, 0, nx*nz*sizeof(float));
                cudaMemset(P_past, 0, nx*nz*sizeof(float));
                cudaMemset(F_pre, 0, nx*nz*sizeof(float));
                cudaMemset(F_past, 0, nx*nz*sizeof(float));

                cudaMemset(psixcz, 0, nx*nz*sizeof(float));
                cudaMemset(psixcx, 0, nx*nz*sizeof(float));
                cudaMemset(psizcz, 0, nx*nz*sizeof(float));
                cudaMemset(psizdx, 0, nx*nz*sizeof(float));

                cudaMemset(atenxcx, 0, nx*nz*sizeof(float));
                cudaMemset(atenxcz, 0, nx*nz*sizeof(float));
                cudaMemset(atenzdx, 0, nx*nz*sizeof(float));
                cudaMemset(atenzcz, 0, nx*nz*sizeof(float));

                cudaMemset(zitaxcx, 0, nx*nz*sizeof(float));
                cudaMemset(zitaxcz, 0, nx*nz*sizeof(float));
                cudaMemset(zitazdx, 0, nx*nz*sizeof(float));
                cudaMemset(zitazcz, 0, nx*nz*sizeof(float));

             
                for (it=0;it<nt;it++){

                    launchBackPropagation(stream1, stream2, nx, nz, nt, &PF_h[(nt-it-1)*nx*nz*2], P_pre, P_past, F_pre, F_past, aux1, aux2, aux3, aux4, v_hor, v_po, v_nmo, p2cx, p2cz, r2dx, r2cz, dh2, P_pre2,
                    F_pre2, a_x, b_x, a_z, b_z, psixcx, psixcz, psizdx, psizcz, dh, CPML, zitaxcx, zitaxcz, zitazdx, zitazcz, atenxcx, atenxcz, atenzdx, atenzcz, Res, C, PF, it, sx, sz, dt,
                    gcx, gcz, gdx);
                    
                    float *URSS = P_past;
                    P_past = P_pre;
                    P_pre = URSS;
            
                    float *URSS2 = F_past;
                    F_past = F_pre;
                    F_pre = URSS2;
                                
                }
                // field=fopen("field3.bin", "wb");
                // fwrite(PF_h2,sizeof(float),nx*nz*nt, field);
                // fclose(field);
            }

            MPI_Reduce(&fiTotal,&phi_function[n],1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
            
            if(rank==0){
                for(int counter=1;counter<rsize;counter++){
                    MPI_Send(&phi_function[n], 1,MPI_FLOAT, counter, 0, MPI_COMM_WORLD);
                }

            }
            else{
                MPI_Recv(&phi_function[n], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            cudaMemcpy(gcx_h, gcx, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(gcz_h, gcz, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(gdx_h, gdx, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);

            MPI_Reduce(gcx_h, gcx_h_m, nx*nz, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(gcz_h, gcz_h_m, nx*nz, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(gdx_h, gdx_h_m, nx*nz, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);


            if(rank==0){
                printf("%s", "Error total : ");
                printf("%f\n", phi_function[n]);
                memcpy(gcx_h, gcx_h_m, nx*nz*sizeof(float));
                memcpy(gcz_h ,gcz_h_m, nx*nz*sizeof(float));
                memcpy(gdx_h ,gdx_h_m, nx*nz*sizeof(float));
                for(int counter=1;counter<rsize;counter++){
                    MPI_Send(gcx_h_m, nx*nz,MPI_FLOAT, counter, 0, MPI_COMM_WORLD);
                    MPI_Send(gcz_h_m, nx*nz,MPI_FLOAT, counter, 0, MPI_COMM_WORLD);
                    MPI_Send(gdx_h_m, nx*nz,MPI_FLOAT, counter, 0, MPI_COMM_WORLD);
                }
            }else{
                MPI_Recv(gcx_h, nx*nz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(gcz_h, nx*nz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(gdx_h, nx*nz ,MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            fiTotal = 0;
            cudaMemcpy(gcx, gcx_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gcz, gcz_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gdx, gdx_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
            if(n<it_LBFGS){

                if (n == it_LBFGS-1){
                    cudaMemcpy(back_vnmo, v_nmo, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(back_vpo, v_po, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(back_vhor, v_hor, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(back_gcz, gcz, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(back_gcx, gcx, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(back_gdx, gdx, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                }


                float ngcx = 0, ngcz = 0, ngdx = 0; 
                for(int i = 0; i<nx*nz; i++){
                    ngcx += powf(gcx_h[i],2);
                    ngcz += powf(gcz_h[i],2);
                    ngdx += powf(gdx_h[i],2);
                }
                ngcx = sqrtf(ngcx);
                ngcz = sqrtf(ngcz);
                ngdx = sqrtf(ngdx);

                NORMAL_h(gcx, gcz, gdx, nx, nz, ngcx, ngcz, ngdx);

                // cudaMemcpy(gcx_h, gcx, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                // cudaMemcpy(gcz_h, gcz, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                // cudaMemcpy(gdx_h, gdx, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);

                UpdateModel_h(gcx, gcz, gdx, v_hor, v_po, v_nmo, 3e5, 2e5, 2.5e5, nx, nz);
            }
            else if(n>=it_LBFGS){
                cudaMemcpy(q_gcz, gcz, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(q_gcx, gcx, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(q_gdx, gdx, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                compute_sy_h(s_vp, s_vn, s_vh, y_gcz, y_gdx, y_gcx, v_po, v_nmo, v_hor, back_vpo, back_vnmo, back_vhor, gcz, gdx, gcx, back_gcz, back_gdx, back_gcx, nx, nz, k);
                for (int l = k; l>=0; l--){
                    
                    cudaMemcpy(norm1_h, &y_gcz[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(norm2_h, &s_vp[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_sigma_v = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_sigma_v += (norm1_h[i]*norm2_h[i])/2.0;

                    cudaMemcpy(norm1_h, q_gcz, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_vp = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_vp += (norm1_h[i]*norm2_h[i])/2.0;

                    sigma_v[l] = 1.0/temp_sigma_v;
                    epsilon_v[l] = sigma_v[l]*temp_vp;



                    cudaMemcpy(norm1_h, &y_gdx[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(norm2_h, &s_vn[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_sigma_d = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_sigma_d += (norm1_h[i]*norm2_h[i])/2.0;

                    cudaMemcpy(norm1_h, q_gdx, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_vn = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_vn += (norm1_h[i]*norm2_h[i])/2.0;

                    sigma_d[l] = 1.0/temp_sigma_d;
                    epsilon_d[l] = sigma_d[l]*temp_vn;


                    cudaMemcpy(norm1_h, &y_gcx[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(norm2_h, &s_vh[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_sigma_e = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_sigma_e += (norm1_h[i]*norm2_h[i])/2.0;

                    cudaMemcpy(norm1_h, q_gcx, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_vh = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_vh += (norm1_h[i]*norm2_h[i])/2.0;

                    sigma_e[l] = 1.0/temp_sigma_e;
                    epsilon_e[l] = sigma_e[l]*temp_vh;

                    compute_qi_h(q_gcz, q_gdx, q_gcx, y_gcz, y_gdx, y_gcx, epsilon_v[l], epsilon_d[l], epsilon_e[l], nx, nz, l);
                }

                cudaMemcpy(norm1_h, &s_vp[k*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(norm2_h, &y_gcz[k*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                float temp_sy_v = 0;
                float temp_yy_v = 0;
                for(int i = 0; i<nx*nz; i++){
                    temp_sy_v += (norm1_h[i]*norm2_h[i])/2.0;
                    temp_yy_v += powf(norm2_h[i],2)/2.0;
                }
                gamma_v = temp_sy_v/temp_yy_v;

                cudaMemcpy(norm1_h, &s_vn[k*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(norm2_h, &y_gdx[k*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                float temp_sy_d = 0;
                float temp_yy_d = 0;
                for(int i = 0; i<nx*nz; i++){
                    temp_sy_d += (norm1_h[i]*norm2_h[i])/2.0;
                    temp_yy_d += powf(norm2_h[i],2)/2.0;
                }
                gamma_d = temp_sy_d/temp_yy_d;

                cudaMemcpy(norm1_h, &s_vh[k*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(norm2_h, &y_gcx[k*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                float temp_sy_e = 0;
                float temp_yy_e = 0;
                for(int i = 0; i<nx*nz; i++){
                    temp_sy_e += (norm1_h[i]*norm2_h[i])/2.0;
                    temp_yy_e += powf(norm2_h[i],2)/2.0;
                }

                gamma_e = temp_sy_e/temp_yy_e;

                compute_ri_h(r_b1, r_b2, r_b3, q_gcz, q_gdx, q_gcx, gamma_v, gamma_d, gamma_e, nx, nz);

                for (int l=0;l<=k;l++){
                    cudaMemcpy(norm1_h, &y_gcz[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(norm2_h, r_b1, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_vp = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_vp += (norm1_h[i]*norm2_h[i])/2.0;
                    beta_v = sigma_v[l]*temp_vp;

                    cudaMemcpy(norm1_h, &y_gdx[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(norm2_h, r_b2, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_vn= 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_vn += (norm1_h[i]*norm2_h[i])/2.0;
                    beta_d = sigma_d[l]*temp_vn;

                    cudaMemcpy(norm1_h, &y_gcx[l*nx*nz], nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(norm2_h, r_b3, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
                    float temp_vh = 0;
                    for(int i = 0; i<nx*nz; i++)
                        temp_vh += (norm1_h[i]*norm2_h[i])/2.0;
                    beta_e = sigma_e[l]*temp_vh;

                    update_r_h(r_b1, r_b2, r_b3, s_vp, s_vn, s_vh, epsilon_v[l], epsilon_d[l], epsilon_e[l], beta_v, beta_d, beta_e, nx, nz, l);


                }
                cudaMemcpy(back_vnmo, v_nmo, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(back_vpo, v_po, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(back_vhor, v_hor, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(back_gcz, gcz, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(back_gcx, gcx, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(back_gdx, gdx, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);

                float temp = 1e10;
                while (temp > phi_function[n]){

                    update_LBFGS_h(r_b1, r_b2, r_b3, v_po, v_nmo, v_hor, back_vpo, back_vnmo, back_vhor, nx, nz, alpha_v_LBFGS, alpha_d_LBFGS, alpha_e_LBFGS, 1, 1, 1, 0, 0);
                    fiTotal = 0;
                    for(int i = rank; i<=nshots; i+=rsize){

                        cudaMemset(psi_px, 0, nx*nz*sizeof(float));
                        cudaMemset(psi_fz, 0, nx*nz*sizeof(float));
                        cudaMemset(aten_p, 0, nx*nz*sizeof(float));
                        cudaMemset(aten_f, 0, nx*nz*sizeof(float));
                        cudaMemset(z_px, 0, nx*nz*sizeof(float));
                        cudaMemset(z_fz, 0, nx*nz*sizeof(float));
                        cudaMemset(P_pre, 0, nx*nz*sizeof(float));
                        cudaMemset(P_past, 0, nx*nz*sizeof(float));
                        cudaMemset(F_pre, 0, nx*nz*sizeof(float));
                        cudaMemset(F_past, 0, nx*nz*sizeof(float));


                        for (it=0;it<nt;it++){
                            PSI_F_h(P_pre, F_pre, a_x, b_x, a_z, b_z, psi_px, psi_fz, nx, nz, dh, CPML);
                            SecondDerivate_P_h(P_pre, F_pre, p2, r2, dh2, nx, nz);
                            ZETA_F_h(p2, r2, P_pre, F_pre, a_x, b_x, a_z, b_z, psi_px, psi_fz, z_px, z_fz, aten_p, aten_f, nx, nz, dh, dh2, CPML);
                            propagator_F_h(p2, r2, PF, P_pre, F_pre, P_past, F_past, v_hor, v_po, v_nmo, aten_p, aten_f, source, shot2, 
                            C, dh2, nx, nz, posShots[i], sz, it, CPML);

                            float *URSS = P_past;
                            P_past = P_pre;
                            P_pre = URSS;
                    
                            float *URSS2 = F_past;
                            F_past = F_pre;
                            F_pre = URSS2;
                        }
                
                        res_h(Res, &shot1[nx*nt*i], shot2, nx, nt);
                        cudaMemcpy(residual_h, Res, nx*nt*sizeof(float), cudaMemcpyDeviceToHost);
                 
                        float fi = 0;
                        
                        for(int j = 0; j<(nx*nt); j++){
                            fi += powf(residual_h[j],2);
                        }
                        fi = fi/2;
                        fiTotal += fi;
                        // printf("%f\n", fiTotal);
                        sx +=space;
                    }
                    temp=0;
                    MPI_Reduce(&fiTotal,&temp,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                
                    if(rank==0){
                        for(int counter=1;counter<rsize;counter++){
                            MPI_Send(&temp, 1,MPI_FLOAT, counter, 0, MPI_COMM_WORLD);
                        }

                    }
                    else{
                        MPI_Recv(&temp, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }

                    // printf("%f\n", temp);

                    if (isnan(temp)){
                        temp = 1e30;
                        count = count +1;
                        alpha_v_LBFGS = alpha_v_LBFGS/2.0;
                        alpha_d_LBFGS = alpha_d_LBFGS/2.0;
                        alpha_e_LBFGS = alpha_e_LBFGS/2.0;
                                
                        if (count <= 5)
                            if(rank==0)
                                printf("new alpha = %f ....\n",alpha_v_LBFGS);
                    }

                    else if (temp > phi_function[n]){
                        alpha_v_LBFGS = alpha_v_LBFGS/2.0;
                        alpha_d_LBFGS = alpha_d_LBFGS/2.0;
                        alpha_e_LBFGS = alpha_e_LBFGS/2.0;
                        count = count + 1;
                                
                        if (count <= 5){
                            if(rank==0)
                                printf("new alpha = %f .... \n",alpha_v_LBFGS);
                        }
                    }

                    if (count > 5){
                        cudaMemcpy(v_po, back_vpo, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                        cudaMemcpy(v_nmo, back_vnmo, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                        cudaMemcpy(v_hor, back_vhor, nx*nz*sizeof(float), cudaMemcpyDeviceToDevice);
                                
                        n = iterations;
                        break;
                    }

                }

                if (k < m-1){k = k+1;}
            }

            if (sw == 1){
                cudaMemcpy(s_vp, s_vp+(nx*nz), nx*nz*(m-1)*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(s_vn, s_vn+(nx*nz), nx*nz*(m-1)*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(s_vh, s_vh+(nx*nz), nx*nz*(m-1)*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(y_gcz, y_gcz+(nx*nz), nx*nz*(m-1)*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(y_gdx, y_gdx+(nx*nz), nx*nz*(m-1)*sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(y_gcx, y_gcx+(nx*nz), nx*nz*(m-1)*sizeof(float), cudaMemcpyDeviceToDevice);
            }

            if (k == m-1){sw = 1;}
        }
        f+=3;
        cudaMemcpy(gcx_h, v_hor, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gcz_h, v_po, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gdx_h, v_nmo, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
    }

    end = clock();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    if(rank==0)
        printf("time: %e\n", time_spent);


    

    // cudaMemcpy(gcx_h, v_hor, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(gcz_h, v_po, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(gdx_h, v_nmo, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);

    // for(int j = 0; j < nx*nz; j++)
    //     gcz_h[j] = sqrtf(gcz_h[j]);

    if(rank==0){
        field=fopen("vh_ini.bin", "wb");
        fwrite(gcx_h,sizeof(float),nx*nz, field);
        fclose(field);

        field=fopen("vp_ini.bin", "wb");
        fwrite(gcz_h,sizeof(float),nx*nz, field);
        fclose(field);

        field=fopen("vn_ini.bin", "wb");
        fwrite(gdx_h,sizeof(float),nx*nz, field);
        fclose(field);
    }

    

    MPI_Finalize();
    return 0;
}
