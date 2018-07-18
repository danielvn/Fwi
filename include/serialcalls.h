#ifndef SERIALCALLS_H
#define SERIALCALLS_H
# define PI 3.141592653589793
# define TILE_WIDTH_X 512
# define I(ix,iz) (ix)+nx*(iz)

#ifdef  __cplusplus
  
    extern "C" {
    	void pre_variables_h (float *vel, float *epsilon, float *delta, float *v_hor, float *v_po, float *v_nmo, int nx, int nz);
		void CPML_x_h(float *a_x, float *b_x, int CPML, float Vmax, int nx, float dt, float dh, float f, float d0, float L);
		void CPML_z_h(float *a_z, float *b_z, int CPML, float Vmax, int nz, float dt, float dh, float f, float d0, float L);
		void PSI_F_h (float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, int nx, int nz, float dh, int CPML);
		void ZETA_F_h (float *p2, float *r2, float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float *z_px, float *z_fz, float *aten_p, float *aten_f, int nx, int nz, float dh, float dh2, int CPML);
		void propagator_F_h (float *p2, float *r2, float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *v_hor, float *v_po, float *v_nmo, float *aten_p, float *aten_f, float *source, float *shot, float C, float dh2, int nx, int nz, int sx, int sz, int it, int CPML);
		void res_h(float* Res, float *shot1, float *shot2, int nx, int nz);
		void SecondDerivate_P_h (float *P_pre, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz);
		void launchPropagation(cudaStream_t stream1, cudaStream_t stream2, float *PF_h, float *PF, float *P_past, float *P_pre, float *F_past, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz,
		float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float dh, int CPML, float *z_px, float *z_fz, float *aten_p, float *aten_f, float *v_hor, float *v_po,
		float *v_nmo, float *source, float *shot2, float C, int sx, int sz, int it);
		void launchBackPropagation(cudaStream_t stream1, cudaStream_t stream2, int nx, int nz, int nt, float *PF_h, float *P_pre, float *P_past, float *F_pre, float *F_past, float *aux1,
		float *aux2, float *aux3, float *aux4, float *v_hor, float *v_po, float *v_nmo, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float dh2, float *P_pre2, float *F_pre2,
		float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, float dh, int CPML, float *zitaxcx, float *zitaxcz, float *zitazdx,
		float *zitazcz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz, float *Res, float C, float *PF, int it, int sx, int sz, float dt, float *gcx, float *gcz, float *gdx);
		void NORMAL_h(float *gcx, float *gcz, float *gdx, int nx, int nz, float ngcx, float ngcz, float ngdx);
		void UpdateModel_h(float *gcx, float *gcz, float *gdx, float *v_hor, float *v_po, float *v_nmo, float alpha1,  float alpha2, float alpha3, int nx, int nz);
		void compute_sy_h (float *s_vp, float *s_vn, float *s_vh, float *y_b1, float *y_b2, float *y_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, float *b1, float *b2, float *b3, float *back_b1, float *back_b2, float *back_b3, int nx, int nz, int k);
		void compute_qi_h (float *q_b1, float *q_b2, float *q_b3, float *y_b1, float *y_b2, float *y_b3, float epsilon_v, float epsilon_d, float epsilon_e, int nx, int nz, int k);
		void compute_ri_h (float *r_b1, float *r_b2, float *r_b3, float *q_b1, float *q_b2, float *q_b3, float gamma_v, float gamma_d, float gamma_e, int nx, int nz);
		void update_r_h (float *r_b1, float *r_b2, float *r_b3, float *s_vp, float *s_vn, float *s_vh, float epsilon_v, float epsilon_d, float epsilon_e, float beta_v, float beta_d, float beta_e, int nx, int nz, int k);
		void update_LBFGS_h (float *r_b1, float *r_b2, float *r_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, int nx, int nz, float alpha_v, float alpha_d, float alpha_e, int flag_v, int flag_d, int flag_e, int flag_min, int flag_max);
    }
#else
		void pre_variables_h (float *vel, float *epsilon, float *delta, float *v_hor, float *v_po, float *v_nmo, int nx, int nz);
		void CPML_x_h(float *a_x, float *b_x, int CPML, float Vmax, int nx, float dt, float dh, float f, float d0, float L);
		void CPML_z_h(float *a_z, float *b_z, int CPML, float Vmax, int nz, float dt, float dh, float f, float d0, float L);
		void PSI_F_h (float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, int nx, int nz, float dh, int CPML);
		void ZETA_F_h (float *p2, float *r2, float *P, float *F, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float *z_px, float *z_fz, float *aten_p, float *aten_f, int nx, int nz, float dh, float dh2, int CPML);
		void propagator_F_h (float *p2, float *r2, float *PF, float *P_pre, float *F_pre, float *P_past, float *F_past, float *v_hor, float *v_po, float *v_nmo, float *aten_p, float *aten_f, float *source, float *shot, float C, float dh2, int nx, int nz, int sx, int sz, int it, int CPML);
		void res_h(float* Res, float *shot1, float *shot2, int nx, int nz);
		void SecondDerivate_P_h (float *P_pre, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz);
		void launchPropagation(cudaStream_t stream1, cudaStream_t stream2, float *PF_h, float *PF, float *P_past, float *P_pre, float *F_past, float *F_pre, float *p2, float *r2, float dh2, int nx, int nz,
		float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_fz, float dh, int CPML, float *z_px, float *z_fz, float *aten_p, float *aten_f, float *v_hor, float *v_po,
		float *v_nmo, float *source, float *shot2, float C, int sx, int sz, int it);
		void launchBackPropagation(cudaStream_t stream1, cudaStream_t stream2, int nx, int nz, int nt, float *PF_h, float *P_pre, float *P_past, float *F_pre, float *F_past, float *aux1,
		float *aux2, float *aux3, float *aux4, float *v_hor, float *v_po, float *v_nmo, float *p2cx, float *p2cz, float *r2dx, float *r2cz, float dh2, float *P_pre2, float *F_pre2,
		float *a_x, float *b_x, float *a_z, float *b_z, float *psixcx, float *psixcz, float *psizdx, float *psizcz, float dh, int CPML, float *zitaxcx, float *zitaxcz, float *zitazdx,
		float *zitazcz, float *atenxcx, float *atenxcz, float *atenzdx, float *atenzcz, float *Res, float C, float *PF, int it, int sx, int sz, float dt, float *gcx, float *gcz, float *gdx);
		void NORMAL_h(float *gcx, float *gcz, float *gdx, int nx, int nz, float ngcx, float ngcz, float ngdx);
		void UpdateModel_h(float *gcx, float *gcz, float *gdx, float *v_hor, float *v_po, float *v_nmo, float alpha1,  float alpha2, float alpha3, int nx, int nz);
		void compute_sy_h (float *s_vp, float *s_vn, float *s_vh, float *y_b1, float *y_b2, float *y_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, float *b1, float *b2, float *b3, float *back_b1, float *back_b2, float *back_b3, int nx, int nz, int k);
		void compute_qi_h (float *q_b1, float *q_b2, float *q_b3, float *y_b1, float *y_b2, float *y_b3, float epsilon_v, float epsilon_d, float epsilon_e, int nx, int nz, int k);
		void compute_ri_h (float *r_b1, float *r_b2, float *r_b3, float *q_b1, float *q_b2, float *q_b3, float gamma_v, float gamma_d, float gamma_e, int nx, int nz);
		void update_r_h (float *r_b1, float *r_b2, float *r_b3, float *s_vp, float *s_vn, float *s_vh, float epsilon_v, float epsilon_d, float epsilon_e, float beta_v, float beta_d, float beta_e, int nx, int nz, int k);
		void update_LBFGS_h (float *r_b1, float *r_b2, float *r_b3, float *vp, float *vn, float *vh, float *back_vp, float *back_vn, float *back_vh, int nx, int nz, float alpha_v, float alpha_d, float alpha_e, int flag_v, int flag_d, int flag_e, int flag_min, int flag_max);
#endif
#endif