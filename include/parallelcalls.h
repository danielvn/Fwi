#ifndef PARALLELCALLS_H
#define PARALLELCALLS_H
# define PI 3.141592653589793
# define TILE_WIDTH_X 512
# define I(ix,iz) (ix)+nx*(iz)
# define layer 15
# define min_v 1.8e6
# define max_v 3.5e6
# define min_d 1.8e6
# define max_d 4e6
# define min_e 1.8e6
# define max_e 4.5e6

#ifdef  __cplusplus
	extern "C" {
		__global__ void pre_variables (float *vel, float *epsilon, float *delta, float *v_hor, float *v_po, float *v_nmo, int nx, int nz);
		__global__ void CPML_x(float *a_x, float *b_x, int CPML, float Vmax, int nx, float dt, float dh, float f, float d0, float L);
		__global__ void CPML_z(float *a_z, float *b_z, int CPML, float Vmax, int nz, float dt, float dh, float f, float d0, float L);
		__global__ void PSI_F (float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, int nx, int nz, float dh, int CPML);
		__global__ void ZETA_F (float *p2, float *r2, float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float *z_px, float *z_fz, float *aten_p, float *aten_f, int nx, int nz, float dh, float dh2, int CPML);
		__global__ void propagator_F (float *p2, float *r2, float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *v_hor, float *v_po, float *v_nmo, float *aten_p, float *aten_f, float *source, float *shot, float C, float dh2, int nx, int nz, int sx, int sz, int it, int CPML);
		__global__ void res(float* Res, float *shot1, float *shot2, int nx, int nz);
		__global__ void SecondDerivate_P (float *P_pre, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz);
		__global__ void retroPropagation(float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *res, float C, float dh2, int nx, int nz, int nt, int sz, int it, int CPML, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz);
		__global__ void entryPoint(float *P_pre, float *F_pre, float *aux1, float *aux2, float *aux3, float *aux4, float *v_hor, float *v_po, float *v_nmo, int nx, int nz);
		__global__ void PSI_R (float *aux1, float *aux2, float *aux3, float *aux4, float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, int nx, int nz, float dh, int CPML);
		__global__ void secondDerivate(float *aux1, float *aux2, float *aux3, float *aux4, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float dh2, int nx, int nz);
		__global__ void ZETA_R (float *p2cx, float *p2cz, float *r2dx, float *r2cz, float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, float *zitaxcx, float *zitaxcz, float *zitazdx, float *zitazcz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz, int nx, int nz, float dh, float dh2, int CPML);
		__global__ void GRADIENT(float *P_pre, float *F_pre, float *P_past, float *F_past, float *gcx, float *gcz, float *gdx, float dh2, float dt, int nx, int nz);
		__global__ void NORMAL(float *gcx, float *gcz, float *gdx, int nx, int nz, float ngcx, float ngcz, float ngdx);
		__global__ void UpdateModel(float *gcx, float *gcz, float *gdx, float *v_hor, float *v_po, float *v_nmo, float alpha1,  float alpha2, float alpha3, int nx, int nz);
		__global__ void compute_sy (float *s_vp, float *s_vn, float *s_vh, float *y_b1, float *y_b2, float *y_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, float *b1, float *b2, float *b3, float *back_b1, float *back_b2, float *back_b3, int nx, int nz, int k);
		__global__ void compute_qi (float *q_b1, float *q_b2, float *q_b3, float *y_b1, float *y_b2, float *y_b3, float epsilon_v, float epsilon_d, float epsilon_e, int nx, int nz, int k);
		__global__ void compute_ri (float *r_b1, float *r_b2, float *r_b3, float *q_b1, float *q_b2, float *q_b3, float gamma_v, float gamma_d, float gamma_e, int nx, int nz);
		__global__ void update_r (float *r_b1, float *r_b2, float *r_b3, float *s_vp, float *s_vn, float *s_vh, float epsilon_v, float epsilon_d, float epsilon_e, float beta_v, float beta_d, float beta_e, int nx, int nz, int k);
		__global__ void update_LBFGS (float *r_b1, float *r_b2, float *r_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, int nx, int nz, float alpha_v, float alpha_d, float alpha_e, int flag_v, int flag_d, int flag_e, int flag_min, int flag_max);
	}
#endif
#endif