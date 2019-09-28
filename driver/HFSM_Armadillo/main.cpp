/** \brief Shell Model Calculation in Hartree Fock Basis with Angular Momentum Projection included
 *
 * \version 0.65
 * \author S.M. Wang (Wang Simin) and S.J. Dai (Dai Sijie)
 * \brief Copyright (c) 2014, Peking University. Written by S.M. Wang and S.J. Dai, wangsimin89@gmail.com
 *
 */

#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<cmath>
#include<complex>
#include<numeric>
#include<cstdio>
#include<cstdlib>
#include<string>
#include<bitset>
#include<limits>
#include<algorithm>
#include<bitset>
#include<vector>
#include<ctime>
#include <gsl/gsl_math.h>
#include <gsl/gsl_combination.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_chebyshev.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <armadillo>
#include <omp.h>
#define ARMA_USE_ARPACK

using namespace std;
using namespace arma;


/***********************************************************************************************************************/

double Jacobi (double x, int n, double a, double b)
{
    if (n==0)
    {
        return 1.0;
    }
    else if (n==1)
    {
        return  0.5 * (a - b + (a + b + 2.0)*x);
    }
    else
    {
        double p0, p1, a1, a2, a3, a4, p2=0.0;
        int i;
        p0 = 1.0;
        p1 = 0.5 * (a - b + (a + b + 2)*x);

        for(i=1; i<n; ++i)
        {
            a1 = 2.0*(i+1.0)*(i+a+b+1.0)*(2.0*i+a+b);
            a2 = (2.0*i+a+b+1.0)*(a*a-b*b);
            a3 = (2.0*i+a+b)*(2.0*i+a+b+1.0)*(2.0*i+a+b+2.0);
            a4 = 2.0*(i+a)*(i+b)*(2.0*i+a+b+2.0);
            p2 = 1.0/a1*((a2 + a3*x)*p1 - a4*p0);

            p0 = p1;
            p1 = p2;
        }

        return p2;
    }
}

class Wigner_Function
{
public:
	double Wigner_d_Matrix (double j, double m1, double m2, double beta);
	cx_double Wigner_D_Matrix (double j, double m1, double m2, double alpha, double beta, double gamma);
};

double Wigner_Function::Wigner_d_Matrix (double j, double m1, double m2, double beta)
{
    int k = fabs(m1)>fabs(m2)?int(j-fabs(m1)+0.5):int(j-fabs(m2)+0.5);
    int a = int(fabs(m1-m2)+0.5);
    int b = int(2*(j-k)-a+0.5);
    int sign = m1>m2?((a&1)?-1:1):1;
    double half_beta = beta/2;

    return sign*gsl_sf_pow_int(sin(half_beta),a)*gsl_sf_pow_int(cos(half_beta),b)*sqrt(gsl_sf_choose(2*j-k,k+a)/double(gsl_sf_choose(k+b,b)))*Jacobi(cos(beta),k,a,b);
}

inline cx_double Wigner_Function::Wigner_D_Matrix (double j, double m1, double m2, double alpha, double beta, double gamma)
{
    return exp(cx_double(0,-m2*gamma-m1*alpha))*Wigner_d_Matrix(j,m1,m2,beta);
}

double YredME(int L,int l1,double j1,int l2,double j2)
/** \brief Calculates the reduced matrix element
 *
 * \return <(l1 1/2) j1 || Y_L || (l2 1/2) j2> * sqrt(4*pi)
 *
 */
{
    int xj1 = int(j1*2+0.5);
    int xj2 = int(j2*2+0.5);

    int LL=(L<<1);
    int ll1=(l1<<1);
    int ll2=(l2<<1);

    double fact=(L+((xj2+1)>>1))&1?-1:1;
    fact=fact*sqrt((ll1+1)*(ll2+1.)*(LL+1.)*(xj1+1)*(xj2+1));

    return fact*gsl_sf_coupling_6j(ll1,xj1,1,xj2,ll2,LL)*gsl_sf_coupling_3j(ll1,LL,ll2,0,0,0);
}

double HORadInt(int L,int n1,int l1,int n2,int l2)
/** \brief computes radial integral R(n1,l1, alpha*r) R(n2,l2, alpha*r) r^(L+2), where R's are h.o. radial functions
 *
 * \return integral of R(n1,l1, alpha*r) R(n2,l2, alpha*r) r^(L+2)
 *
 */
{
    double lnprefact,lnsum;

    if( (l1+l2+L)%2 != 0) 	  // must be even overall
	{
        return 0.0;
    }

    lnprefact = (gsl_sf_lnfact(n1)+gsl_sf_lnfact(n2) + log(2.)*(n1+n2-L)- gsl_sf_lndoublefact(2*n1+2*l1+1) - gsl_sf_lndoublefact(2*n2+2*l2+1))*0.5 + gsl_sf_lnfact((l2-l1+L)>>1) + gsl_sf_lnfact((l1-l2+L)>>1);

    int qmax = min(n1,n2);
    int qmin = max(0,max(n1-((l2-l1+L)>>1),n2-((l1-l2+L)>>1)));
    double sum = 0.0;
    for (int q = qmin;q<=qmax;q++)
	{
	    lnsum =  gsl_sf_lndoublefact(l1+l2+L+2*q+1) -q*log(2.) - gsl_sf_lnfact(q) - gsl_sf_lnfact(n1-q) -gsl_sf_lnfact(n2-q) - gsl_sf_lnfact(q+(l1-l2+L)/2-n2) - gsl_sf_lnfact(q+(l2-l1+L)/2-n1);

        sum = sum + exp( lnprefact+lnsum);
	}

    if(abs(n1-n2)&1) sum = -sum;

    return sum;
}

void intrinsic(vec & q_diag, double& q1,double& q2)
/** \brief Calculates the intrinsic values for the quadrupole operator as defined in PRC 61, 034303 (2000)
 *
 * \return quadrupole operator
 *
 */
{
      double fact = sqrt(5.0*M_1_PI);

      q1=q_diag(2)*fact*0.25;

      q2=(fact*q_diag(0)*0.5+q1)*sqrt(2.);
      q2/=(2.*sqrt(3.));
}

template<typename T1, typename T2>
cx_double compute_trace(const T1 & a,const T2 & b)
{
    return accu(a%b.st());
}

/***********************************************************************************************************************/

class J_Orbit
{
public:
    void set (int node, int orbit, double angular);
    void operator = (J_Orbit & a);
	void show();

    int n;
    int l;
    double j;
};

void J_Orbit::set (int node, int orbit, double angular)
{
    n = node;
    l = orbit;
    j = angular;
}

void J_Orbit::operator = (J_Orbit & a)
{
    n = a.n;
    l = a.l;
    j = a.j;
}

void J_Orbit::show()
{
	cout << j << "  " << l << "  " << n << endl;
}

/***********************************************************************************************************************/

class M_State
{
public:
    void set (int i, int node, int orbit, double angular, double jz, double pn);
    void operator = (M_State & a);
	void show();

    int index;
    int n;
    int l;
    double j;
    double m;
    double Iz;
};

void M_State::set (int i, int node, int orbit, double angular, double jz, double pn)
{
    index = i;
    n = node;
    l = orbit;
    j = angular;
    m = jz;
    Iz = pn;
}

void M_State::operator = (M_State & a)
{
    index = a.index;
    n = a.n;
    l = a.l;
    j = a.j;
    m = a.m;
    Iz = a.Iz;
}

void M_State::show()
{
	cout << index << "  " << Iz << "  " << j << "  " << l << "  " << m << "  " << n << endl;
}
/***********************************************************************************************************************/

class HF_SPS
///interface for shell model by WuQiang
{
public:
	void set(int i, double iso, int par, double jz, double energy);
	int index;
	double m;
	int par;
	double Iz;
	double spe;
};

void HF_SPS::set(int i, double iso, int parity, double jz, double energy)
{
	index = i;
	Iz    = iso;
	par   = parity;
	m     = jz;
	spe   = energy;
}

/***********************************************************************************************************************/

class Coupled_State
{
public:
    void operator = (Coupled_State & a);

    uvec orbit;
    int jmax;
    int jmin;
    vec v;
};

void Coupled_State::operator = (Coupled_State & a)
{
    orbit = a.orbit;
    jmax = a.jmax;
    jmin = a.jmin;
    v = a.v;
}

/***********************************************************************************************************************/

template <typename T>
class Interaction
{
public:
    Interaction();
    Interaction(string space, string ME, int A, double bet);
    template<typename T1>
    Interaction (const Interaction<T1> & Force);
    ~Interaction();

    template<typename T1>
    void operator = (Interaction<T1> & Force);

    void unpack (size_t num, int &i, int &j, int &k, int &l);
    void unpack (size_t num, int &i, int &j);
    void pack (int i, int j, int k, int l, size_t &num);
    void pack (int i, int j, size_t &num);
    void sort_index_M (imat * indx, imat * m_record);

    void Read_Orbit_Iso (string space);
    void Read_Orbit_PN (string space);
    void Read_Orbit_OLS (string space);
	void Read_Orbit_PN_QBox (string space);
    void Read_Interaction_Iso (string ME);
    void Read_Interaction_PN (string ME);
    void Read_Interaction_OLS (string ME);
	void Read_Interaction_PN_QBox (string ME);
    void Construct_Interaction_Iso ();
    void Construct_Interaction_PN ();
    void Construct_Interaction_OLS ();
    void Setup_Uncoupled_States_Iso (J_Orbit * J, int dim, Coupled_State * vtbme);
    void Setup_Uncoupled_States_PN (J_Orbit ** Jstates, int * dim, Coupled_State * vtbme);
//	void Setup_Uncoupled_States_PN_QBox (J_Orbit ** Jstates, int * dim, Coupled_State * vtbme);
    void Setup_Uncoupled_States_OLS (J_Orbit * J, int dim, Coupled_State * vtbme);

    template<typename T1, typename T2, typename T3>
    void Basis_Transform (Interaction<T3> & force, T1 * psil, T2 * psir, vec * m_record = NULL);
    template<typename T1>
    void TBME_MatrixForm_AS (Mat<T1> * tbme_mat, vec * m_record = NULL);
    void TBME_MatrixForm_AS (SpMat<T> * tbme_mat, imat * indx = NULL);
    template<typename T1>
    void TBME_MatrixForm (Mat<T1> * tbme_mat, vec * m_record = NULL);
    void G_Matrix_MemOptimize (Interaction<T> & src, vec * e_sp, vec & e_Fermi, bool axial = false);
    void G_Matrix (Interaction<T> & src, vec * e_sp, vec & e_Fermi, bool axial = false);
    void Creation_Anihilation_Operator ();
    ///void sort_by_rank ();

    static const unsigned int Max_Bit = 128;
    static const int Parity_Domain = 2;
    static const int IsoSpin_Z = 2;
    static const int N_Body = 2;
    static const int FootnoteNum = 2*N_Body;

    unsigned int Dimension_M[IsoSpin_Z];
    int Dimension_J[IsoSpin_Z];
	int **idx_J;
    J_Orbit * J_scheme[IsoSpin_Z];
    M_State * M_scheme[IsoSpin_Z];
    umat index[IsoSpin_Z+1];
    Col<T> TBME[IsoSpin_Z+1];
    Mat<T> OBME[IsoSpin_Z];

    int Mass_Num;
    double hbar_omega;

    vector<bitset<Max_Bit> > Op_OB[IsoSpin_Z];
    vector<bitset<Max_Bit> > Phase_OB[IsoSpin_Z];

    vector<bitset<Max_Bit> > Op_TB[IsoSpin_Z];
    vector<bitset<Max_Bit> > Phase_TB[IsoSpin_Z];
    vector<bitset<Max_Bit> > Op_TB_pn[IsoSpin_Z];
    vector<bitset<Max_Bit> > Phase_TB_pn[IsoSpin_Z];

    Col<T> TBME_rank[(IsoSpin_Z+1)*(N_Body+1)];
    imat index_rank[(IsoSpin_Z+1)*(N_Body+1)];
	double beta;

	void ME_Output_Mscheme();

private:
    Coupled_State * Orbit;
};

template <typename T>
Interaction<T>::Interaction()
{
    Mass_Num = 0;
    hbar_omega = 0;

    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        J_scheme[i]=NULL;
        M_scheme[i]=NULL;
    }

    Orbit = NULL;
	beta = 0.;
}

template <typename T>
template<typename T1>
Interaction<T>::Interaction (const Interaction<T1> & Force)
{
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        Dimension_J[i] = Force.Dimension_J[i];
        Dimension_M[i] = Force.Dimension_M[i];

        J_scheme[i] = new J_Orbit [Dimension_J[i]];
        M_scheme[i] = new M_State [Dimension_M[i]];
        OBME[i].resize(Force.OBME[i].n_rows,Force.OBME[i].n_cols);

        for(int j=Dimension_J[i]-1;j>=0;j--) J_scheme[i][j] = Force.J_scheme[i][j];

        for(int j=Dimension_M[i]-1;j>=0;j--) M_scheme[i][j] = Force.M_scheme[i][j];

        for(int j=OBME[i].n_elem-1;j>=0;j--) OBME[i](j) = Force.OBME[i](j);
    }

    for(int i=IsoSpin_Z;i>=0;i--)
    {
        index[i] = Force.index[i];
        TBME[i].resize(Force.TBME[i].n_elem);

        for(int j=TBME[i].n_elem-1;j>=0;j--) TBME[i](j) = Force.TBME[i](j);
    }

    Mass_Num = Force.Mass_Num;
    hbar_omega = Force.hbar_omega;

    Orbit = NULL;
    Creation_Anihilation_Operator();
	beta = Force.beta;
}

template <typename T>
Interaction<T>::Interaction(string space, string ME, int A, double bet):Mass_Num(A)
{
    int IsIso = true;
	beta = bet;

//        Read_Orbit_OLS(space);
		Read_Orbit_PN_QBox(space);
//		Read_Orbit_PN(space);
		int size_dim = 0;
		for(int i=IsoSpin_Z-1; i>=0; i--)
		{
			int dim = Dimension_J[i]*(Dimension_J[i]+1)/2;
			size_dim += dim*(dim+1)/2;
		}
		size_dim += Dimension_J[0]*Dimension_J[1]*(Dimension_J[0]*Dimension_J[1]+1)/2;
//        cout << size_dim << endl;
		Orbit = new Coupled_State [size_dim];
        Setup_Uncoupled_States_PN(J_scheme,Dimension_J,Orbit);
        Read_Interaction_PN_QBox(ME);
//		Read_Interaction_PN(ME);
        Construct_Interaction_PN();
		cout.flush();
//		ME_Output_Mscheme();


/*    if(IsIso)
    {
        Read_Orbit_Iso(space);
        int dim = Dimension_J[0]*(Dimension_J[0]+1)/2;
        Orbit = new Coupled_State [dim*(dim+1)/2];
        Setup_Uncoupled_States_Iso(J_scheme[0],Dimension_J[0],Orbit);
        Read_Interaction_Iso(ME);
        Construct_Interaction_Iso();
    }
    else
    {
        Read_Orbit_PN(space);

        int size_dim = 0;
        for(int i=IsoSpin_Z-1;i>=0;i--)
        {
            int dim = Dimension_J[i]*(Dimension_J[i]+1)/2;
            size_dim+=dim*(dim+1)/2;
        }
        size_dim+=Dimension_J[0]*Dimension_J[1]*(Dimension_J[0]*Dimension_J[1]+1)/2;
        Orbit = new Coupled_State [size_dim];
        Setup_Uncoupled_States_PN(J_scheme,Dimension_J,Orbit);
        Read_Interaction_PN(ME);
        Construct_Interaction_PN();
    }
*/

/*
        Read_Orbit_Iso(space);
        int size_dim = 0;
        for(int i=IsoSpin_Z-1;i>=0;i--)
        {
            int dim = Dimension_J[i]*(Dimension_J[i]+1)/2;
            size_dim+=dim*(dim+1)/2;
        }
        size_dim+=Dimension_J[0]*Dimension_J[1]*(Dimension_J[0]*Dimension_J[1]+1)/2;
        Orbit = new Coupled_State [size_dim];
        Setup_Uncoupled_States_PN(J_scheme,Dimension_J,Orbit);
        Read_Interaction_PN(ME);
        Construct_Interaction_PN();*/

    Creation_Anihilation_Operator();
}

template <typename T>
Interaction<T>::~Interaction()
{
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        if(J_scheme[i]!=NULL) delete [] J_scheme[i];
        if(M_scheme[i]!=NULL) delete [] M_scheme[i];
    }

    if(Orbit!=NULL) delete [] Orbit;
}

template<typename T>
template<typename T1>
void Interaction<T>::operator = (Interaction<T1> & Force)
{
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        Dimension_J[i] = Force.Dimension_J[i];
        Dimension_M[i] = Force.Dimension_M[i];

        J_scheme[i] = new J_Orbit [Dimension_J[i]];
        M_scheme[i] = new M_State [Dimension_M[i]];
        OBME[i].resize(Force.OBME[i].n_rows,Force.OBME[i].n_cols);

        Op_OB[i] = Force.Op_OB[i];
        Op_TB[i] = Force.Op_TB[i];
        Op_TB_pn[i] = Force.Op_TB_pn[i];
        Phase_OB[i] = Force.Phase_OB[i];
        Phase_TB[i] = Force.Phase_TB[i];
        Phase_TB_pn[i] = Force.Phase_TB_pn[i];

        for(int j=Dimension_J[i]-1;j>=0;j--) J_scheme[i][j] = Force.J_scheme[i][j];

        for(int j=Dimension_M[i]-1;j>=0;j--) M_scheme[i][j] = Force.M_scheme[i][j];

        for(int j=OBME[i].n_elem-1;j>=0;j--) OBME[i](j) = Force.OBME[i](j);
    }

    for(int i=IsoSpin_Z;i>=0;i--)
    {
        index[i] = Force.index[i];
        TBME[i].resize(Force.TBME[i].n_elem);

        for(int j=TBME[i].n_elem-1;j>=0;j--) TBME[i](j) = Force.TBME[i](j);
    }

    Mass_Num = Force.Mass_Num;
    hbar_omega = Force.hbar_omega;

    Orbit = NULL;

    Creation_Anihilation_Operator ();
}

/*
template<typename T>
void Interaction<T>::sort_by_rank ()
{
    for(int i=IsoSpin_Z;i>=0;i--)
    {
        const int dim = Dimension_M[i]*Dimension_M[i];
        const int offset = i*(N_Body+1);

        ivec count_num = zeros<ivec>(N_Body+1);

        index_rank[offset] = zeros<imat>(Dimension_M[i],Dimension_M[i]);
        index_rank[offset+1] = zeros<imat>(dim,Dimension_M[i]);
        index_rank[offset+2] = zeros<imat>(dim,dim);
        for(int j=N_Body;j--;j>=0)
        {
            index_rank[offset+j].fill(-1);
            TBME_rank[offset+j] = zeros<Col<T> >(index[i].n_elem);
        }

        for(int j=index[i].n_elem-1;j>=0;j--)
        {
            uvec col = index[i].col(j);
            int ii = col(0);
            int jj = col(1);
            int kk = col(2);
            int ll = col(3);
            if(ii==kk)
            {
                if(jj==ll)
                {
                    index_rank[offset](ii,jj) = count_num(0);
                    TBME_rank[offset](count_num(0)++) = TBME[i](j);
                }
                else
                {
                    index_rank[offset](jj*Dimension_M[i]+ll,ii) = count_num(1);
                    TBME_rank[offset](count_num(1)++) = TBME[i](j);
                }
            }
            else
            {
                if((ii==ll)||(jj==)
            }
        }
        index[i];
    }
}*/

template<typename T>
void Interaction<T>::Creation_Anihilation_Operator ()
{
    //p, n
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        Op_OB[i].resize(OBME[i].n_elem, bitset<Max_Bit>(0));
        Phase_OB[i].resize(OBME[i].n_elem, bitset<Max_Bit>(0));

        for(int j=OBME[i].n_cols-1;j>=0;j--)
        {
            const int offset = j*OBME[i].n_rows;
            for(int k=OBME[i].n_rows-1;k>=0;k--)
            {
                bitset<Max_Bit> *p = &Op_OB[i][offset+k];
                bitset<Max_Bit> *p_phase = &Phase_OB[i][offset+k];
                p->flip(k);
                p->flip(j);

                int lmin = min(j,k);
                int lmax = max(j,k);
                for(int l=lmin+1;l<lmax;l++)
                {
                    p_phase->flip(l);
                }
            }
        }
    }

    //pp, nn
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        Op_TB[i].resize(index[i].n_cols, bitset<Max_Bit>(0));
        Phase_TB[i].resize(index[i].n_cols, bitset<Max_Bit>(0));

        for(int j=index[i].n_cols-1;j>=0;j--)
        {
            bitset<Max_Bit> *p = &Op_TB[i][j];
            bitset<Max_Bit> *p_phase = &Phase_TB[i][j];
            subview_col<uword> col = index[i].col(j);

            for(int k=FootnoteNum-1;k>=0;k--)
            {
                p->flip(col(k));
            }

            for(unsigned int k=col(3)+1;k<col(2);k++)
            {
                p_phase->flip(k);
            }
            for(unsigned int k=col(1)+1;k<col(0);k++)
            {
                if((k!=col(2))&&(k!=col(3))) p_phase->flip(k);
            }
        }
    }

    //pn
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        Op_TB_pn[i].resize(index[IsoSpin_Z].n_cols, bitset<Max_Bit>(0));
        Phase_TB_pn[i].resize(index[IsoSpin_Z].n_cols, bitset<Max_Bit>(0));

        for(int j=index[IsoSpin_Z].n_cols-1;j>=0;j--)
        {
            bitset<Max_Bit> *p = &Op_TB_pn[i][j];
            bitset<Max_Bit> *p_phase = &Phase_TB_pn[i][j];
            subview_col<uword> col = index[IsoSpin_Z].col(j);

            for(int k=FootnoteNum-IsoSpin_Z+i;k>=0;k-=2)
            {
                p->flip(col(k));
            }

            int kmin = min(col(i),col(i+IsoSpin_Z));
            int kmax = max(col(i),col(i+IsoSpin_Z));
            for(int k=kmin+1;k<kmax;k++)
            {
                p_phase->flip(k);
            }
        }
    }
}

template<typename T>
void Interaction<T>::sort_index_M (imat * indx, imat * m_record)
{
    const int row_dim = 2;

    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        const int dim = Dimension_M[loop]*(Dimension_M[loop]-1)/2;
        indx[loop] = zeros<imat>(row_dim,dim);
        ivec temp = zeros<ivec>(dim);

        #pragma omp parallel for
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            const int offseti = i*(i-1)/2;
            for(int j=i-1;j>=0;j--)
            {
                const int col_indx = offseti+j;
                indx[loop](0,col_indx) = i;
                indx[loop](1,col_indx) = j;

                double d = M_scheme[loop][i].m+M_scheme[loop][j].m;
                temp(col_indx) = d>=0?int(d+0.5):-int(-d+0.5);
            }
        }

        uvec q = sort_index(temp,"ascend");
        ivec unique_m = unique(temp);

        indx[loop] = indx[loop].cols(q);
        temp = temp(q);
        m_record[loop] = zeros<imat>(row_dim,unique_m.n_elem);
        m_record[loop].row(0) = unique_m.st();
        for(int i=unique_m.n_elem-1;i>=0;i--)
        {
            uvec individual = find(temp==unique_m(i));
            m_record[loop](1,i) = individual.n_elem;
        }
    }

    const int dim = Dimension_M[0]*Dimension_M[1];
    indx[IsoSpin_Z] = zeros<imat>(row_dim,dim);
    ivec temp = zeros<ivec>(dim);

    #pragma omp parallel for
    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        const int offseti = i*Dimension_M[1];
        for(int j=Dimension_M[1]-1;j>=0;j--)
        {
            const int col_indx = offseti+j;
            indx[IsoSpin_Z](0,col_indx) = i;
            indx[IsoSpin_Z](1,col_indx) = j;
            temp(col_indx) = int(M_scheme[0][i].m+M_scheme[1][j].m+M_scheme[0][i].j+M_scheme[1][j].j+0.5)-int(M_scheme[0][i].j+M_scheme[1][j].j+0.5);
        }
    }

    uvec q = sort_index(temp,"ascend");
    ivec unique_m = unique(temp);

    indx[IsoSpin_Z] = indx[IsoSpin_Z].cols(q);
    temp = temp(q);
    m_record[IsoSpin_Z] = zeros<imat>(row_dim,unique_m.n_elem);
    m_record[IsoSpin_Z].row(0) = unique_m.st();

    for(int i=unique_m.n_elem-1;i>=0;i--)
    {
        uvec individual = find(temp==unique_m(i));
        m_record[IsoSpin_Z](1,i) = individual.n_elem;
    }
}

template<typename T>
template<typename T1, typename T2, typename T3>
void Interaction<T>::Basis_Transform (Interaction<T3> & force, T1 * psil, T2 * psir, vec * m_record)
{
    const double EPS = 1e-4;

    Mass_Num = force.Mass_Num;
    hbar_omega = force.hbar_omega;

    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        Dimension_M[loop] = psil[loop].n_cols;
        if(m_record!=NULL)
        {
            M_scheme[loop] = new M_State [Dimension_M[loop]];
            for(int i=Dimension_M[loop]-1;i>=0;i--)
            {
                M_scheme[loop][i].m = m_record[loop](i);
            }
        }
    }

//	for(int i=0; i<=psil[0].n_rows-1;i++)
//	{
//		for(int j=0; j<=psil[0].n_cols-1;j++)
//		  cout << psil[0](i,j) << "  ";
//		cout << endl;
//	}

    ///One body matrix elements
//    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
//    {
//        if(Dimension_M[loop]>0) OBME[loop] = psil[loop].st()*force.OBME[loop]*psir[loop];
//        else OBME[loop] = zeros<Mat<T> >(0,0);
//    }
	for(int iso=0; iso<=IsoSpin_Z-1; iso++)
	{
		OBME[iso] = zeros<cx_mat>(psil[iso].n_cols,psir[iso].n_cols);
		cx_mat psilt=psil[iso].st();
		for(int a=0; a<=psilt.n_rows-1; a++)
		  for(int b=0; b<=psir[iso].n_cols-1; b++)
			for(int i=0; i<=psilt.n_cols-1; i++)
			  for(int j=0; j<=psir[iso].n_rows-1; j++)
				OBME[iso](a,b)+= psilt(a,i)*force.OBME[iso](i,j)*psir[iso](j,b);
	}

	cout << OBME[0](0,0) << endl;

    ///Two body matrix elements
    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        const int dim_New = Dimension_M[loop];
        const int dim_Old = force.Dimension_M[loop];

        if((dim_New<=0)||(dim_Old<=0)) continue;

        Cube<T> tbme_cube= zeros<Cube<T> >(dim_Old,dim_Old,dim_Old*dim_Old);
        if(dim_Old>0)
        {
            #pragma omp parallel for
            for(int n=force.index[loop].n_cols-1;n>=0;n--)
            {
                subview_col<uword> col = force.index[loop].col(n);
                tbme_cube(col(2),col(3),col(0)*dim_Old+col(1)) = force.TBME[loop](n);
            }
        }

        Cube<T> tbme_cube_bk = zeros<Cube<T> >(dim_New,dim_New,dim_Old*dim_Old);
        #pragma omp parallel for
        for(int n=tbme_cube.n_slices-1;n>=0;n--)
        {
            tbme_cube_bk.slice(n) = psir[loop].st()*tbme_cube.slice(n)*psir[loop];
            tbme_cube_bk.slice(n) -= tbme_cube_bk.slice(n).st();
        }

        Cube<T> tbme_cube2= zeros<Cube<T> >(dim_Old,dim_Old,dim_New*(dim_New-1)/2);
        #pragma omp parallel for
        for(int k=dim_New-1;k>=0;k--)
        {
            const int offsetk = k*(k-1)/2;
            for(int l=k-1;l>=0;l--)
            {
                for(int i=dim_Old-1;i>=0;i--)
                {
                    const int offseti = i*dim_Old;
                    for(int j=dim_Old-1;j>=0;j--)
                    {
                        T d = tbme_cube_bk(k,l,offseti+j);
                        if(abs(d)>EPS)
                        {
                            tbme_cube2(i,j,offsetk+l) = d;
                        }
                    }
                }
            }
        }

        Cube<T> tbme_cube2_bk= zeros<Cube<T> >(dim_New,dim_New,dim_New*(dim_New-1)/2);
        #pragma omp parallel for
        for(int n=tbme_cube2.n_slices-1;n>=0;n--)
        {
            tbme_cube2_bk.slice(n) = psil[loop].st()*tbme_cube2.slice(n)*psil[loop];
            tbme_cube2_bk.slice(n) -= tbme_cube2_bk.slice(n).st();

            for(int i=dim_New-1;i>=0;i--)
            {
                for(int j=i;j>=0;j--)
                {
                    tbme_cube2_bk(j,i,n) = 0;
                }
            }
        }

        uvec indx = find(abs(tbme_cube2_bk)>EPS);
        index[loop] = zeros<umat>(FootnoteNum,indx.n_elem);
        TBME[loop] = zeros<Col<T> >(indx.n_elem);

        int count_num = 0;
        for(int k=dim_New-1;k>=0;k--)
        {
            const int offsetk = k*(k-1)/2;
            for(int l=k-1;l>=0;l--)
            {
                for(int i=dim_New-1;i>=0;i--)
                {
                    for(int j=i-1;j>=0;j--)
                    {
                        T d = tbme_cube2_bk(i,j,offsetk+l);

                        if(abs(d)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;
                            index[loop].col(count_num) = col;
                            TBME[loop](count_num++) = d;
                        }
                    }
                }
            }
        }
    }

    const int dim_HF0 = Dimension_M[0];
    const int dim_HF1 = Dimension_M[1];
    const int pair_dim = Dimension_M[0]*Dimension_M[1];
    const int dim_HF0_Old = force.Dimension_M[0];
    const int dim_HF1_Old = force.Dimension_M[1];
    const int pair_dim_Old = force.Dimension_M[0]*force.Dimension_M[1];

    if((pair_dim_Old>0)&&(pair_dim>0))
    {

        Cube<T> tbme_cube= zeros<Cube<T> >(dim_HF0_Old,dim_HF1_Old,pair_dim_Old);

        #pragma omp parallel for
        for(int n=force.index[IsoSpin_Z].n_cols-1;n>=0;n--)
        {
            subview_col<uword> col = force.index[IsoSpin_Z].col(n);
            tbme_cube(col(2),col(3),col(0)*dim_HF1_Old+col(1)) = force.TBME[IsoSpin_Z](n);
        }

        Cube<T> tbme_cube_bk= zeros<Cube<T> >(dim_HF0,dim_HF1,pair_dim_Old);
        #pragma omp parallel for
        for(int n=pair_dim_Old-1;n>=0;n--)
        {
            tbme_cube_bk.slice(n) = psir[0].st()*tbme_cube.slice(n)*psir[1];
        }

        Cube<T> tbme_cube2= zeros<Cube<T> >(dim_HF0_Old,dim_HF1_Old,pair_dim);
        #pragma omp parallel for
        for(int n=tbme_cube_bk.n_slices-1;n>=0;n--)
        {
            const int k=n/dim_HF1_Old;
            const int l=n%dim_HF1_Old;
            for(int i=dim_HF0-1;i>=0;i--)
            {
                const int offseti = i*dim_HF1;
                for(int j=dim_HF1-1;j>=0;j--)
                {
                    T d = tbme_cube_bk(i,j,n);
                    if(abs(d)>EPS)
                    {
                        tbme_cube2(k,l,offseti+j) = d;
                    }
                }
            }
        }

        Cube<T> tbme_cube2_bk= zeros<Cube<T> >(dim_HF0,dim_HF1,pair_dim);
        #pragma omp parallel for
        for(int n=pair_dim-1;n>=0;n--)
        {
            tbme_cube2_bk.slice(n) = psil[0].st()*tbme_cube2.slice(n)*psil[1];
        }

        uvec indx = find(abs(tbme_cube2_bk)>EPS);
        index[IsoSpin_Z] = zeros<umat>(FootnoteNum,indx.n_elem);
        TBME[IsoSpin_Z] = zeros<Col<T> >(indx.n_elem);

        int count_num = 0;
        for(int i=dim_HF0-1;i>=0;i--)
        {
            for(int j=dim_HF1-1;j>=0;j--)
            {
                for(int k=dim_HF0-1;k>=0;k--)
                {
                    const int offsetk = k*dim_HF1;
                    for(int l=dim_HF1-1;l>=0;l--)
                    {
                        T d = tbme_cube2_bk(i,j,offsetk+l);

                        if(abs(d)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;

                            index[IsoSpin_Z].col(count_num) = col;
                            TBME[IsoSpin_Z](count_num++) = d;
                        }
                    }
                }
            }
        }
    }

    Creation_Anihilation_Operator();
}


template <typename T>
template <typename T1>
void Interaction<T>::TBME_MatrixForm (Mat<T1> * tbme_mat, vec * m_record)
{
    //pp, nn TBME
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        const int dim = Dimension_M[i]*Dimension_M[i];
        tbme_mat[i] = zeros<Mat<T1> >(dim,dim);
        if(m_record!=NULL) m_record[i].resize(dim);

        #pragma omp parallel for
        for(int j=index[i].n_cols-1;j>=0;j--)
        {
            subview_col<uword> col = index[i].col(j);
            int ii=col(0);
            int jj=col(1);
            int kk=col(2);
            int ll=col(3);

            tbme_mat[i](ii*Dimension_M[i]+jj,kk*Dimension_M[i]+ll) = TBME[i](j);
            tbme_mat[i](jj*Dimension_M[i]+ii,kk*Dimension_M[i]+ll) = -TBME[i](j);
            tbme_mat[i](ii*Dimension_M[i]+jj,ll*Dimension_M[i]+kk) = -TBME[i](j);
            tbme_mat[i](jj*Dimension_M[i]+ii,ll*Dimension_M[i]+kk) = TBME[i](j);
            if(m_record!=NULL) m_record[i](ii*Dimension_M[i]+jj) = M_scheme[i][ii].m+M_scheme[i][jj].m;
        }
    }

    //pn TBME
    const int dim = Dimension_M[0]*Dimension_M[1];
    tbme_mat[IsoSpin_Z] = zeros<Mat<T1> >(dim,dim);
    if(m_record!=NULL) m_record[IsoSpin_Z].resize(dim);

    #pragma omp parallel for
    for(int j=index[IsoSpin_Z].n_cols-1;j>=0;j--)
    {
        subview_col<uword> col = index[IsoSpin_Z].col(j);
        int ii=col(0);
        int jj=col(1);
        int kk=col(2);
        int ll=col(3);

        tbme_mat[IsoSpin_Z](ii*Dimension_M[1]+jj,kk*Dimension_M[1]+ll) = TBME[IsoSpin_Z](j);
        if(m_record!=NULL) m_record[IsoSpin_Z](ii*Dimension_M[1]+jj) = M_scheme[0][ii].m+M_scheme[1][jj].m;
    }
}


template <typename T>
void Interaction<T>::TBME_MatrixForm_AS (SpMat<T> * tbme_mat, imat * indx)
{
    if(indx!=NULL)
    {
        //pp, nn TBME
        for(int i=IsoSpin_Z-1;i>=0;i--)
        {
            Col<size_t> indx_bk(indx[i].n_cols);

            #pragma omp parallel for
            for(int j=indx[i].n_cols-1;j>=0;j--)
            {
                size_t num;
                pack(indx[i](0,j),indx[i](1,j),num);
                indx_bk(j) = num;
            }

            umat location(2,index[i].n_cols);

            #pragma omp parallel for
            for(int j=index[i].n_cols-1;j>=0;j--)
            {
                subview_col<uword> col = index[i].col(j);
                int ii=col(0);
                int jj=col(1);
                int kk=col(2);
                int ll=col(3);

                size_t num1,num2;

                pack(ii,jj,num1);
                pack(kk,ll,num2);

                uvec q1 = find(indx_bk==num1);
                uvec q2 = find(indx_bk==num2);

                location(0,j) = q1(0);
                location(1,j) = q2(0);
            }
            const int dim = Dimension_M[i]*(Dimension_M[i]-1)/2;
            tbme_mat[i] = SpMat<T>(location,TBME[i],dim,dim);
        }

        //pn TBME
        Col<size_t> indx_bk(indx[IsoSpin_Z].n_cols);

        #pragma omp parallel for
        for(int j=indx[IsoSpin_Z].n_cols-1;j>=0;j--)
        {
            size_t num;
            pack(indx[IsoSpin_Z](0,j),indx[IsoSpin_Z](1,j),num);
            indx_bk(j) = num;
        }

        umat location(2,index[IsoSpin_Z].n_cols);

        #pragma omp parallel for
        for(int j=index[IsoSpin_Z].n_cols-1;j>=0;j--)
        {
            subview_col<uword> col = index[IsoSpin_Z].col(j);
            int ii=col(0);
            int jj=col(1);
            int kk=col(2);
            int ll=col(3);

            size_t num1,num2;

            pack(ii,jj,num1);
            pack(kk,ll,num2);

            uvec q1 = find(indx_bk==num1);
            uvec q2 = find(indx_bk==num2);

            location(0,j) = q1(0);
            location(1,j) = q2(0);
        }
        const int dim = Dimension_M[0]*Dimension_M[1];
        tbme_mat[IsoSpin_Z] = SpMat<T>(location,TBME[IsoSpin_Z],dim,dim);
    }
    else
    {
        //pp, nn TBME
        for(int i=IsoSpin_Z-1;i>=0;i--)
        {
            umat location(2,index[i].n_cols);

            #pragma omp parallel for
            for(int j=index[i].n_cols-1;j>=0;j--)
            {
                subview_col<uword> col = index[i].col(j);
                int ii=col(0);
                int jj=col(1);
                int kk=col(2);
                int ll=col(3);

                location(0,j) = ii*(ii-1)/2+jj;
                location(1,j) = kk*(kk-1)/2+ll;
            }
            const int dim = Dimension_M[i]*(Dimension_M[i]-1)/2;
            tbme_mat[i] = SpMat<T>(location,TBME[i],dim,dim);
        }

        //pn TBME
        umat location(2,index[IsoSpin_Z].n_cols);

        #pragma omp parallel for
        for(int j=index[IsoSpin_Z].n_cols-1;j>=0;j--)
        {
            subview_col<uword> col = index[IsoSpin_Z].col(j);
            int ii=col(0);
            int jj=col(1);
            int kk=col(2);
            int ll=col(3);

            location(0,j) = ii*Dimension_M[1]+jj;
            location(1,j) = kk*Dimension_M[1]+ll;
        }
        const int dim = Dimension_M[0]*Dimension_M[1];
        tbme_mat[IsoSpin_Z] = SpMat<T>(location,TBME[IsoSpin_Z],dim,dim);
    }
}


template <typename T>
template <typename T1>
void Interaction<T>::TBME_MatrixForm_AS (Mat<T1> * tbme_mat, vec * m_record)
{
    //pp, nn TBME
    for(int i=IsoSpin_Z-1;i>=0;i--)
    {
        const int dim = Dimension_M[i]*(Dimension_M[i]-1)/2;
        tbme_mat[i] = zeros<Mat<T1> >(dim,dim);
        if(m_record!=NULL) m_record[i].resize(dim);

        #pragma omp parallel for
        for(int j=index[i].n_cols-1;j>=0;j--)
        {
            subview_col<uword> col = index[i].col(j);
            int ii=col(0);
            int jj=col(1);
            int kk=col(2);
            int ll=col(3);

            tbme_mat[i](ii*(ii-1)/2+jj,kk*(kk-1)/2+ll) = TBME[i](j);
            if(m_record!=NULL) m_record[i](ii*(ii-1)/2+jj) = M_scheme[i][ii].m+M_scheme[i][jj].m;
        }
    }

    //pn TBME
    const int dim = Dimension_M[0]*Dimension_M[1];
    tbme_mat[IsoSpin_Z] = zeros<Mat<T1> >(dim,dim);
    if(m_record!=NULL) m_record[IsoSpin_Z].resize(dim);

    #pragma omp parallel for
    for(int j=index[IsoSpin_Z].n_cols-1;j>=0;j--)
    {
        subview_col<uword> col = index[IsoSpin_Z].col(j);
        int ii=col(0);
        int jj=col(1);
        int kk=col(2);
        int ll=col(3);

        tbme_mat[IsoSpin_Z](ii*Dimension_M[1]+jj,kk*Dimension_M[1]+ll) = TBME[IsoSpin_Z](j);
        if(m_record!=NULL) m_record[IsoSpin_Z](ii*Dimension_M[1]+jj) = M_scheme[0][ii].m+M_scheme[1][jj].m;
    }
}

template <typename T>
void Interaction<T>::Read_Orbit_Iso (string space)
/** \brief Read J orbit in isospin representation
 *
 * \return J_Orbit
 *
 */
{
    int jdim;

    ifstream ifile(space.c_str());

    for(int j=IsoSpin_Z-1;j>=0;j--)
    {
        ifile>>jdim;
        Dimension_J[j] = jdim;
        J_scheme[j] = new J_Orbit [jdim];

        int count_num = 0;
        for(int i=0;i<jdim;i++)
        {
            ifile>>J_scheme[j][i].n;
            ifile>>J_scheme[j][i].l;
            ifile>>J_scheme[j][i].j;

            count_num += int(2*J_scheme[j][i].j+1.5);
        }
        M_scheme[j] = new M_State [count_num];
        Dimension_M[j] = count_num;

        count_num = 0;
        for(int i=0;i<jdim;i++)
        {
            int degeneracy = int(2*J_scheme[j][i].j+1.5);
            for(int k=0;k<degeneracy;k++)
            {
                M_scheme[j][count_num].index = i;
                M_scheme[j][count_num].n = J_scheme[j][i].n;
                M_scheme[j][count_num].l = J_scheme[j][i].l;
                M_scheme[j][count_num].j = J_scheme[j][i].j;
                M_scheme[j][count_num].m = k-J_scheme[j][i].j;
                M_scheme[j][count_num].Iz = j-0.5;

                count_num++;
            }
        }
    }
    ifile.close();
}

template <typename T>
void Interaction<T>::Read_Orbit_PN (string space)
/** \brief Read J orbit in proton-neutron representation
 *
 * \return J_Orbit
 *
 */
{
    double length;

    ifstream ifile(space.c_str());
    string s;

    do
    {
        ifile>>s;
    }
    while(s!="energy:");

    ifile>>length;
    ifile>>hbar_omega;

    do
    {
        ifile>>s;
    }
    while(s!="orbits");

    int dim;
    ifile>>dim;
    int jdim = (dim>>1);

    for(int j=IsoSpin_Z-1;j>=0;j--)
    {
        Dimension_J[j] = jdim;
        J_scheme[j] = new J_Orbit [jdim];
    }

    int indx,n,l,j2,tz;
    int count_num_J[IsoSpin_Z] = {0,0};
    int count_num_M[IsoSpin_Z] = {0,0};
    for(int i=0;i<dim;i++)
    {
        do
        {
            ifile>>s;
        }
        while(s!="Number:");
        ifile>>indx;
        ifile>>n;
        ifile>>l;
        ifile>>j2;
        ifile>>tz;

        tz++;
        tz>>=1;

        J_scheme[tz][count_num_J[tz]].n = n;
        J_scheme[tz][count_num_J[tz]].l = l;
        J_scheme[tz][count_num_J[tz]].j = double(j2)*0.5;

        count_num_J[tz]++;
        count_num_M[tz]+= j2+1;
    }
    ifile.close();

    for(int j=IsoSpin_Z-1;j>=0;j--)
    {
        M_scheme[j] = new M_State [count_num_M[j]];
        Dimension_M[j] = count_num_M[j];

        int count_num = 0;
        for(int i=0;i<Dimension_J[j];i++)
        {
            int degeneracy = int(2*J_scheme[j][i].j+1.5);
            for(int k=0;k<degeneracy;k++)
            {
                M_scheme[j][count_num].index = i;
                M_scheme[j][count_num].n = J_scheme[j][i].n;
                M_scheme[j][count_num].l = J_scheme[j][i].l;
                M_scheme[j][count_num].j = J_scheme[j][i].j;
                M_scheme[j][count_num].m = k-J_scheme[j][i].j;
                M_scheme[j][count_num].Iz = j-0.5;

                count_num++;
            }
        }
    }
}

template <typename T>
void Interaction<T>::Read_Orbit_OLS (string space)
/** \brief Read J orbit in isospin representation
 *
 * \return J_Orbit
 *
 */
{
    int jdim;

    ifstream ifile(space.c_str());
    string s;

    do
    {
        ifile>>s;
    }
    while(s!="STATES=");

    ifile>>jdim;

    do
    {
        ifile>>s;
    }
    while(s!="j");

    for(int j=IsoSpin_Z-1;j>=0;j--)
    {
        Dimension_J[j] = jdim;
        J_scheme[j] = new J_Orbit [jdim];
    }

    int count_num = 0;
    int indx;
    for(int i=0;i<jdim;i++)
    {
        ifile>>indx;
        ifile>>indx;
        ifile>>J_scheme[0][i].n;
        ifile>>J_scheme[0][i].l;
        ifile>>J_scheme[0][i].j;

        J_scheme[1][i].n=J_scheme[0][i].n;
        J_scheme[1][i].l=J_scheme[0][i].l;
        J_scheme[1][i].j=J_scheme[0][i].j;

        count_num += int(2*J_scheme[0][i].j+1.5);
    }

    do
    {
        ifile>>s;
    }
    while(s!="HBAR*OMEGA=");

    ifile>>hbar_omega;

    for(int j=IsoSpin_Z-1;j>=0;j--)
    {
        M_scheme[j] = new M_State [count_num];
        Dimension_M[j] = count_num;
    }

    for(int j=IsoSpin_Z-1;j>=0;j--)
    {
        count_num = 0;
        for(int i=0;i<jdim;i++)
        {
            int degeneracy = int(2*J_scheme[j][i].j+1.5);
            for(int k=0;k<degeneracy;k++)
            {
                M_scheme[j][count_num].index = i;
                M_scheme[j][count_num].n = J_scheme[j][i].n;
                M_scheme[j][count_num].l = J_scheme[j][i].l;
                M_scheme[j][count_num].j = J_scheme[j][i].j;
                M_scheme[j][count_num].m = k-J_scheme[j][i].j;
                M_scheme[j][count_num].Iz = j-0.5;

                count_num++;
            }
        }
    }
    ifile.close();
}

template <typename T>
void Interaction<T>::Read_Orbit_PN_QBox (string space)
{
    ifstream ifile(space.c_str());
	string stemp;
	int jdim, idx_tmp, n, l, j2, tz2, n2_l;
	double spe, evalence;
	string par_hol, in_out;
	J_Orbit *jstmp[2];
	int counter[2] = {0, 0};
	getline(ifile, stemp);
	getline(ifile, stemp);
	for(int i=0; i<5; i++) ifile >> stemp;
	ifile >> hbar_omega;
	for(int i=0; i<6; i++) getline(ifile,stemp);
	for(int i=0; i<5; i++) ifile >> stemp;
	ifile >> jdim;
	jstmp[0] = new J_Orbit[jdim/2];
	jstmp[1] = new J_Orbit[jdim/2];
	idx_J = new int *[2];
	idx_J[0] = new int[jdim/2];
	idx_J[1] = new int[jdim/2];
	getline(ifile,stemp);
	getline(ifile,stemp);
	for(int i=0; i<jdim; i++)
	{
		ifile >> stemp;
		ifile >> idx_tmp;
		ifile >> n;
		ifile >> l;
		ifile >> j2;
		ifile >> tz2;
		ifile >> n2_l;
		ifile >> spe;
		ifile >> evalence;
		ifile >> par_hol;
		ifile >> in_out;
		if(in_out == "outside") continue;
		else
		{
			int isospin = (tz2+1)/2;
			jstmp[isospin][counter[isospin]].n = n;
			jstmp[isospin][counter[isospin]].l = l;
			jstmp[isospin][counter[isospin]].j = j2/2.;
			idx_J[isospin][counter[isospin]]=idx_tmp;
			counter[isospin]++;
		}
	}
    int mcount;
    for(int i=0; i<=1; i++)
    {
        mcount = 0;
        J_scheme[i] = new J_Orbit[counter[i]];
        for(int j=0; j<counter[i]; j++)
        {
            J_scheme[i][j] = jstmp[i][j];
            mcount += int(2*J_scheme[i][j].j+1.5);
        }
        Dimension_J[i] = counter[i];
        M_scheme[i] = new M_State [mcount];
        Dimension_M[i] = mcount;

		mcount = 0;
        for(int j=0;j<counter[i];j++)
        {
            int degeneracy = int(2*J_scheme[i][j].j+1.5);
            for(int k=0; k<degeneracy; k++)
            {
                M_scheme[i][mcount].index = j;
                M_scheme[i][mcount].n = J_scheme[i][j].n;
                M_scheme[i][mcount].l = J_scheme[i][j].l;
                M_scheme[i][mcount].j = J_scheme[i][j].j;
                M_scheme[i][mcount].m = k-J_scheme[i][j].j;
                M_scheme[i][mcount].Iz = i-0.5;

                mcount++;
            }
        }
    }
    ifile.close();
	cout << "Read orbit complete." << endl;
	cout.flush();
/*	for(int i=0; i<=1; i++)
	{
		for(int j=0; j<counter[i]; j++)
			J_scheme[i][j].show();
	    for(int j=0; j<Dimension_M[i]; j++)
			M_scheme[i][j].show();
	}
	cin >> stemp;*/
}



template <typename T>
void Interaction<T>::Setup_Uncoupled_States_Iso (J_Orbit * Jstates, int dim, Coupled_State * vtbme)
/** \brief Setup TBME and uncoupled states in isospin representation
 *
 * \param J_Orbit
 * \return TBME space
 *
 */
{
    for(int i=dim-1;i>=0;i--)
    {
        double ja = Jstates[i].j;
        for(int j=i;j>=0;j--)
        {
            double jb = Jstates[j].j;
            const int offset1 = i*(i+1)/2+j;
            for(int k=i;k>=0;k--)
            {
                double jc = Jstates[k].j;

                int lstart;
                if(i==k) lstart=j;
                else lstart=k;

                for(int l=lstart;l>=0;l--)
                {
                    double jd = Jstates[l].j;
                    const int offset2 = k*(k+1)/2+l;
                    const int indx = offset1*(offset1+1)/2+offset2;
                    uvec col;
                    col<<i<<j<<k<<l<<endr;

                    vtbme[indx].orbit = col;
                    vtbme[indx].jmax = min(int(ja+jb+0.5),int(jc+jd+0.5));
                    vtbme[indx].jmin = max(int(fabs(ja-jb)+0.5),int(fabs(jc-jd)+0.5));
                    if(vtbme[indx].jmax>=vtbme[indx].jmin) vtbme[indx].v = zeros<vec>((vtbme[indx].jmax-vtbme[indx].jmin+1)*IsoSpin_Z);
                }
            }
        }
    }
}

template <typename T>
void Interaction<T>::Setup_Uncoupled_States_PN (J_Orbit ** Jstates, int * dim, Coupled_State * vtbme)
/** \brief Setup TBME and uncoupled states in proton-neutron representation
 *
 * \param J_Orbit
 * \return TBME space
 *
 */
{
    int offset = 0;
    for(int loop=0;loop<IsoSpin_Z;loop++)
    {
        for(int i=dim[loop]-1;i>=0;i--)
        {
            double ja = Jstates[loop][i].j;
            for(int j=i;j>=0;j--)
            {
                double jb = Jstates[loop][j].j;
                const int offset1 = i*(i+1)/2+j;
                for(int k=i;k>=0;k--)
                {
                    double jc = Jstates[loop][k].j;

                    int lstart;
                    if(i==k) lstart=j;
                    else lstart=k;

                    for(int l=lstart;l>=0;l--)
                    {
                        double jd = Jstates[loop][l].j;
                        const int offset2 = k*(k+1)/2+l;
                        const int indx = offset1*(offset1+1)/2+offset2+offset;
                        uvec col;
                        col<<i<<j<<k<<l<<endr;

                        vtbme[indx].orbit = col;
                        vtbme[indx].jmax = min(int(ja+jb+0.5),int(jc+jd+0.5));
                        vtbme[indx].jmin = max(int(fabs(ja-jb)+0.5),int(fabs(jc-jd)+0.5));
                        if(vtbme[indx].jmax>=vtbme[indx].jmin) vtbme[indx].v = zeros<vec>(vtbme[indx].jmax-vtbme[indx].jmin+1);
//						else cout << indx << "  " << ja << "  " << jb << "  " << jc << "  " << jd << endl;
                    }
                }
            }
        }

        int temp =dim[loop]*(dim[loop]+1)/2;
        offset+=temp*(temp+1)/2;
//		cout << offset << endl;
//		string stemp;
//		cin >> stemp;
    }


    for(int i=dim[0]-1;i>=0;i--)
    {
        double ja = Jstates[0][i].j;
        for(int j=dim[1]-1;j>=0;j--)
        {
            double jb = Jstates[1][j].j;
            const int offset1 = i*dim[1]+j;
            for(int k=i;k>=0;k--)
            {
                double jc = Jstates[0][k].j;

                int lstart;
                if(i==k) lstart=j;
                else lstart=dim[1]-1;

                for(int l=lstart;l>=0;l--)
                {
                    double jd = Jstates[1][l].j;
                    const int offset2 = k*dim[1]+l;
                    const int indx = offset1*(offset1+1)/2+offset2+offset;
                    uvec col;
                    col<<i<<j<<k<<l<<endr;

                    vtbme[indx].orbit = col;
                    vtbme[indx].jmax = min(int(ja+jb+0.5),int(jc+jd+0.5));
                    vtbme[indx].jmin = max(int(fabs(ja-jb)+0.5),int(fabs(jc-jd)+0.5));
                    if(vtbme[indx].jmax>=vtbme[indx].jmin) vtbme[indx].v = zeros<vec>(vtbme[indx].jmax-vtbme[indx].jmin+1);
                }
            }
        }
    }

}

//template <typename T>
//void Interaction<T>::Setup_Uncoupled_States_PN_QBox (J_Orbit ** Jstates, int * dim, Coupled_State * vtbme)
//{
//
//}

template <typename T>
void Interaction<T>::Setup_Uncoupled_States_OLS (J_Orbit * Jstates, int dim, Coupled_State * vtbme)
/** \brief Setup TBME and uncoupled states in proton-neutron representation
 *
 * \param J_Orbit
 * \return TBME space
 *
 */
{

    for(int i=dim-1;i>=0;i--)
    {
        double ja = Jstates[i].j;
        for(int j=i;j>=0;j--)
        {
            double jb = Jstates[j].j;
            const int offset1 = i*(i+1)/2+j;
            for(int k=i;k>=0;k--)
            {
                double jc = Jstates[k].j;

                int lstart;
                if(i==k) lstart=j;
                else lstart=k;

                for(int l=lstart;l>=0;l--)
                {
                    double jd = Jstates[l].j;
                    const int offset2 = k*(k+1)/2+l;
                    const int indx = offset1*(offset1+1)/2+offset2;
                    uvec col;
                    col<<i<<j<<k<<l<<endr;

                    vtbme[indx].orbit = col;
                    vtbme[indx].jmax = min(int(ja+jb+0.5),int(jc+jd+0.5));
                    vtbme[indx].jmin = max(int(fabs(ja-jb)+0.5),int(fabs(jc-jd)+0.5));
                    if(vtbme[indx].jmax>=vtbme[indx].jmin) vtbme[indx].v = zeros<vec>((vtbme[indx].jmax-vtbme[indx].jmin+1)*IsoSpin_Z*IsoSpin_Z*IsoSpin_Z);
                }
            }
        }
    }
}

template <typename T>
void Interaction<T>::Construct_Interaction_Iso ()
/** \brief Construct interaction in isospin representation
 *
 * \param TBME J scheme
 * \return TBME M scheme
 *
 */
{
    const double EPS = 1e-4;

    ///pp, nn interaction
    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        const int pair_dim = Dimension_M[loop]*(Dimension_M[loop]-1)/2;
        const int dim = pair_dim*pair_dim;

        umat index_duplicate = zeros<umat>(FootnoteNum,dim);
        Col<T> TBME_duplicate = zeros<Col<T> >(dim);

        int count_num = 0;
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            int oa = M_scheme[loop][i].index;
            int la = M_scheme[loop][i].l;
            int ta = (M_scheme[loop][i].Iz>=0)?1:-1;
            int ja = int(M_scheme[loop][i].j*2+0.5);
            int ma = int(M_scheme[loop][i].m*2+0.5+ja)-ja;
            for(int j=i-1;j>=0;j--)
            {
                int ob = M_scheme[loop][j].index;
                int lb = M_scheme[loop][j].l;
                int tb = (M_scheme[loop][j].Iz>=0)?1:-1;
                int jb = int(M_scheme[loop][j].j*2+0.5);
                int mb = int(M_scheme[loop][j].m*2+0.5+jb)-jb;

                const int offset1 = (oa>=ob)?(oa*(oa+1)/2+ob):(ob*(ob+1)/2+oa);

                for(int k=Dimension_M[loop]-1;k>=0;k--)
                {
                    int oc = M_scheme[loop][k].index;
                    int lc = M_scheme[loop][k].l;
                    int tc = (M_scheme[loop][k].Iz>=0)?1:-1;
                    int jc = int(M_scheme[loop][k].j*2+0.5);
                    int mc = int(M_scheme[loop][k].m*2+0.5+jc)-jc;

                    for(int l=k-1;l>=0;l--)
                    {
                        int od = M_scheme[loop][l].index;
                        int ld = M_scheme[loop][l].l;
                        int td = (M_scheme[loop][l].Iz>=0)?1:-1;
                        int jd = int(M_scheme[loop][l].j*2+0.5);
                        int md = int(M_scheme[loop][l].m*2+0.5+jd)-jd;

                        int tij = ta+tb;
                        int tkl = tc+td;
                        int mij = ma+mb;
                        int mkl = mc+md;
                        int pij = (la+lb)&1;
                        int pkl = (lc+ld)&1;

                        const int offset2 = (oc>=od)?(oc*(oc+1)/2+od):(od*(od+1)/2+oc);
                        const int indx = (offset1>=offset2)?(offset1*(offset1+1)/2+offset2):(offset2*(offset2+1)/2+offset1);

                        const int offset = Orbit[indx].jmax-Orbit[indx].jmin+1;

                        if((mij==mkl)&&(tij==tkl)&&(pij==pkl))
                        {
                            double fact = (((jb-ja+jd-jc)/2)&1)?-1.0:1.0;
                            if(oa==ob) fact*=sqrt(2.0);
                            if(oc==od) fact*=sqrt(2.0);

                            int fact1;
                            double sum = 0.0;
                            for(int jr=max(abs(mij/2),Orbit[indx].jmin);jr<=Orbit[indx].jmax;jr++)
                            {
                                const int jr2 = (jr<<1);

                                fact1 = 1.0;
                                if(oa<ob)
                                {
                                    int sign = ((((ja+jb)>>1)-jr)&1)?-1:1;
                                    fact1 *= -sign;
                                }
                                if(oc<od)
                                {
                                    int sign = ((((jc+jd)>>1)-jr)&1)?-1:1;
                                    fact1 *= -sign;
                                }

                                sum += double(jr2+1)*gsl_sf_coupling_3j(ja,jb,jr2,ma,mb,-mij)*gsl_sf_coupling_3j(jc,jd,jr2,mc,md,-mkl)*Orbit[indx].v(jr-Orbit[indx].jmin+offset)*fact1;

                            }
                            sum *= fact;

                            if(fabs(sum)>EPS)
                            {
                                uvec col;
                                col<<i<<j<<k<<l<<endr;

                                index_duplicate.col(count_num) = col;
                                TBME_duplicate(count_num) = sum;

                                count_num++;
                            }
                        }
                    }
                }
            }
        }

        index[loop]=index_duplicate.cols(0,count_num-1);
        TBME[loop]=TBME_duplicate.subvec(0,count_num-1);
    }


    ///pn interaction
    const int pair_dim = Dimension_M[0]*Dimension_M[1];
    const int dim = pair_dim*pair_dim;

    umat index_duplicate = zeros<umat>(FootnoteNum,dim);
    Col<T> TBME_duplicate = zeros<Col<T> >(dim);

    int count_num = 0;
    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        int oa = M_scheme[0][i].index;
        int la = M_scheme[0][i].l;
        int ta = (M_scheme[0][i].Iz>=0)?1:-1;
        int ja = int(M_scheme[0][i].j*2+0.5);
        int ma = int(M_scheme[0][i].m*2+0.5+ja)-ja;
        for(int j=Dimension_M[1]-1;j>=0;j--)
        {
            int ob = M_scheme[1][j].index;
            int lb = M_scheme[1][j].l;
            int tb = (M_scheme[1][j].Iz>=0)?1:-1;
            int jb = int(M_scheme[1][j].j*2+0.5);
            int mb = int(M_scheme[1][j].m*2+0.5+jb)-jb;

            const int offset1 = (oa>=ob)?(oa*(oa+1)/2+ob):(ob*(ob+1)/2+oa);

            for(int k=Dimension_M[0]-1;k>=0;k--)
            {
                int oc = M_scheme[0][k].index;
                int lc = M_scheme[0][k].l;
                int tc = (M_scheme[0][k].Iz>=0)?1:-1;
                int jc = int(M_scheme[0][k].j*2+0.5);
                int mc = int(M_scheme[0][k].m*2+0.5+jc)-jc;

                for(int l=Dimension_M[1]-1;l>=0;l--)
                {
                    int od = M_scheme[1][l].index;
                    int ld = M_scheme[1][l].l;
                    int td = (M_scheme[1][l].Iz>=0)?1:-1;
                    int jd = int(M_scheme[1][l].j*2+0.5);
                    int md = int(M_scheme[1][l].m*2+0.5+jd)-jd;

                    int tij = ta+tb;
                    int tkl = tc+td;
                    int mij = ma+mb;
                    int mkl = mc+md;
                    int pij = (la+lb)&1;
                    int pkl = (lc+ld)&1;

                    const int offset2 = (oc>=od)?(oc*(oc+1)/2+od):(od*(od+1)/2+oc);
                    const int indx = (offset1>=offset2)?(offset1*(offset1+1)/2+offset2):(offset2*(offset2+1)/2+offset1);

                    const int offset = Orbit[indx].jmax-Orbit[indx].jmin+1;

                    if((mij==mkl)&&(tij==tkl)&&(pij==pkl))
                    {
                        double fact = (((jb-ja+jd-jc)/2)&1)?-1.0:1.0;
                        if(oa==ob) fact*=sqrt(2.0);
                        if(oc==od) fact*=sqrt(2.0);

                        int fact0,fact1;
                        double sum = 0.0;
                        for(int jr=max(abs(mij/2),Orbit[indx].jmin);jr<=Orbit[indx].jmax;jr++)
                        {
                            const int jr2 = (jr<<1);

                            fact0 = 1.0;
                            fact1 = 1.0;
                            if(oa<ob)
                            {
                                int sign = ((((ja+jb)>>1)-jr)&1)?-1:1;
                                fact0 *= sign;
                                fact1 *= -sign;
                            }
                            if(oc<od)
                            {
                                int sign = ((((jc+jd)>>1)-jr)&1)?-1:1;
                                fact0 *= sign;
                                fact1 *= -sign;
                            }

                            double temp = double(jr2+1)*gsl_sf_coupling_3j(ja,jb,jr2,ma,mb,-mij)*gsl_sf_coupling_3j(jc,jd,jr2,mc,md,-mkl);
                            sum += temp*fact0*Orbit[indx].v(jr-Orbit[indx].jmin)*gsl_sf_coupling_3j(1,1,0,ta,tb,-tij)*gsl_sf_coupling_3j(1,1,0,tc,td,-tkl);
                            sum += 3*temp*fact1*Orbit[indx].v(jr-Orbit[indx].jmin+offset)*gsl_sf_coupling_3j(1,1,2,ta,tb,-tij)*gsl_sf_coupling_3j(1,1,2,tc,td,-tkl);
                        }
                        sum*=fact;

                        if(fabs(sum)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;

                            index_duplicate.col(count_num) = col;
                            TBME_duplicate(count_num) = sum;

                            count_num++;
                        }
                    }
                }
            }
        }
    }

    index[IsoSpin_Z] = index_duplicate.cols(0,count_num-1);
    TBME[IsoSpin_Z] = TBME_duplicate.subvec(0,count_num-1);
}

template <typename T>
void Interaction<T>::Construct_Interaction_PN ()
/** \brief Construct interaction in proton-neutron representation
 *
 * \param TBME J scheme
 * \return TBME M scheme
 *
 */
{
    const double EPS = -1e-4;

    ///pp, nn interaction
    int offset = 0;
    for(int loop=0;loop<IsoSpin_Z;loop++)
    {
        const int pair_dim = Dimension_M[loop]*(Dimension_M[loop]-1)/2;
        const int dim = pair_dim*pair_dim;

        umat index_duplicate = zeros<umat>(FootnoteNum,dim);
        Col<T> TBME_duplicate = zeros<Col<T> >(dim);

        int count_num = 0;
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            int oa = M_scheme[loop][i].index;
            int la = M_scheme[loop][i].l;
            int ta = (M_scheme[loop][i].Iz>=0)?1:-1;
            int ja = int(M_scheme[loop][i].j*2+0.5);
            int ma = int(M_scheme[loop][i].m*2+0.5+ja)-ja;
            for(int j=i-1;j>=0;j--)
            {
                int ob = M_scheme[loop][j].index;
                int lb = M_scheme[loop][j].l;
                int tb = (M_scheme[loop][j].Iz>=0)?1:-1;
                int jb = int(M_scheme[loop][j].j*2+0.5);
                int mb = int(M_scheme[loop][j].m*2+0.5+jb)-jb;

                const int offset1 = (oa>=ob)?(oa*(oa+1)/2+ob):(ob*(ob+1)/2+oa);

                for(int k=Dimension_M[loop]-1;k>=0;k--)
                {
                    int oc = M_scheme[loop][k].index;
                    int lc = M_scheme[loop][k].l;
                    int tc = (M_scheme[loop][k].Iz>=0)?1:-1;
                    int jc = int(M_scheme[loop][k].j*2+0.5);
                    int mc = int(M_scheme[loop][k].m*2+0.5+jc)-jc;

                    for(int l=k-1;l>=0;l--)
                    {
                        int od = M_scheme[loop][l].index;
                        int ld = M_scheme[loop][l].l;
                        int td = (M_scheme[loop][l].Iz>=0)?1:-1;
                        int jd = int(M_scheme[loop][l].j*2+0.5);
                        int md = int(M_scheme[loop][l].m*2+0.5+jd)-jd;

                        int tij = ta+tb;
                        int tkl = tc+td;
                        int mij = ma+mb;
                        int mkl = mc+md;
                        int pij = (la+lb)&1;
                        int pkl = (lc+ld)&1;

                        const int offset2 = (oc>=od)?(oc*(oc+1)/2+od):(od*(od+1)/2+oc);
                        const int indx = (offset1>=offset2)?(offset1*(offset1+1)/2+offset2+offset):(offset2*(offset2+1)/2+offset1+offset);

                        if((mij==mkl)&&(tij==tkl)&&(pij==pkl))
                        {
                            double fact = (((jb-ja+jd-jc)/2)&1)?-1.0:1.0;
                            if(oa==ob) fact*=sqrt(2.0);
                            if(oc==od) fact*=sqrt(2.0);

                            int coef;
                            double sum = 0.0;
                            for(int jr=max(abs(mij/2),Orbit[indx].jmin);jr<=Orbit[indx].jmax;jr++)
                            {
                                const int jr2 = (jr<<1);

                                coef = 1.0;
                                if(oa<ob)
                                {
                                    int sign = ((((ja+jb)>>1)-jr)&1)?-1:1;
                                    coef *= -sign;
                                }
                                if(oc<od)
                                {
                                    int sign = ((((jc+jd)>>1)-jr)&1)?-1:1;
                                    coef *= -sign;
                                }
                                sum += double(jr2+1)*gsl_sf_coupling_3j(ja,jb,jr2,ma,mb,-mij)*gsl_sf_coupling_3j(jc,jd,jr2,mc,md,-mkl)*Orbit[indx].v(jr-Orbit[indx].jmin)*coef;

                            }
                            sum *= fact;

                            if(fabs(sum)>EPS)
                            {
                                uvec col;
                                col<<i<<j<<k<<l<<endr;

                                index_duplicate.col(count_num) = col;
                                TBME_duplicate(count_num) = sum;

                                count_num++;
                            }
                        }
                    }
                }
            }
        }

        index[loop]=index_duplicate.cols(0,count_num-1);
        TBME[loop]=TBME_duplicate.subvec(0,count_num-1);

        int temp = Dimension_J[loop]*(Dimension_J[loop]+1)/2;
        offset += temp*(temp+1)/2;
    }


    ///pn interaction
    const int pair_dim = Dimension_M[0]*Dimension_M[1];
    const int dim = pair_dim*pair_dim;

    umat index_duplicate = zeros<umat>(FootnoteNum,dim);
    Col<T> TBME_duplicate = zeros<Col<T> >(dim);

    int count_num = 0;
    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        int oa = M_scheme[0][i].index;
        int la = M_scheme[0][i].l;
        int ta = (M_scheme[0][i].Iz>=0)?1:-1;
        int ja = int(M_scheme[0][i].j*2+0.5);
        int ma = int(M_scheme[0][i].m*2+0.5+ja)-ja;
        for(int j=Dimension_M[1]-1;j>=0;j--)
        {
            int ob = M_scheme[1][j].index;
            int lb = M_scheme[1][j].l;
            int tb = (M_scheme[1][j].Iz>=0)?1:-1;
            int jb = int(M_scheme[1][j].j*2+0.5);
            int mb = int(M_scheme[1][j].m*2+0.5+jb)-jb;

            const int offset1 = oa*Dimension_J[1]+ob;

            for(int k=Dimension_M[0]-1;k>=0;k--)
            {
                int oc = M_scheme[0][k].index;
                int lc = M_scheme[0][k].l;
                int tc = (M_scheme[0][k].Iz>=0)?1:-1;
                int jc = int(M_scheme[0][k].j*2+0.5);
                int mc = int(M_scheme[0][k].m*2+0.5+jc)-jc;

                for(int l=Dimension_M[1]-1;l>=0;l--)
                {
                    int od = M_scheme[1][l].index;
                    int ld = M_scheme[1][l].l;
                    int td = (M_scheme[1][l].Iz>=0)?1:-1;
                    int jd = int(M_scheme[1][l].j*2+0.5);
                    int md = int(M_scheme[1][l].m*2+0.5+jd)-jd;

                    int tij = ta+tb;
                    int tkl = tc+td;
                    int mij = ma+mb;
                    int mkl = mc+md;
                    int pij = (la+lb)&1;
                    int pkl = (lc+ld)&1;

                    const int offset2 = oc*Dimension_J[1]+od;
                    const int indx = (offset1>=offset2)?(offset1*(offset1+1)/2+offset2+offset):(offset2*(offset2+1)/2+offset1+offset);

                    if((mij==mkl)&&(tij==tkl)&&(pij==pkl))
                    {
                        double fact = (((jb-ja+jd-jc)/2)&1)?-1.0:1.0;
                        ///if(oa==ob) fact*=sqrt(2.0);
                        ///if(oc==od) fact*=sqrt(2.0);
                          ///!!!5

                        double sum = 0.0;
                        for(int jr=max(abs(mij/2),Orbit[indx].jmin);jr<=Orbit[indx].jmax;jr++)
                        {
                            const int jr2 = (jr<<1);

                            sum += double(jr2+1)*gsl_sf_coupling_3j(ja,jb,jr2,ma,mb,-mij)*gsl_sf_coupling_3j(jc,jd,jr2,mc,md,-mkl)*Orbit[indx].v(jr-Orbit[indx].jmin);
                        }
                        sum*=fact;

                        if(fabs(sum)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;

                            index_duplicate.col(count_num)=col;
                            TBME_duplicate(count_num)=sum;

                            count_num++;
                        }
                    }
                }
            }
        }
    }

    index[IsoSpin_Z] = index_duplicate.cols(0,count_num-1);
    TBME[IsoSpin_Z] = TBME_duplicate.subvec(0,count_num-1);
}

template <typename T>
void Interaction<T>::Construct_Interaction_OLS ()
/** \brief Construct interaction in isospin representation
 *
 * \param TBME J scheme
 * \return TBME M scheme
 *
 */
{
    const double EPS = 1e-4;

    ///pp, nn interaction
    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        const int pair_dim = Dimension_M[loop]*(Dimension_M[loop]-1)/2;
        const int dim = pair_dim*pair_dim;

        umat index_duplicate = zeros<umat>(FootnoteNum,dim);
        Col<T> TBME_duplicate = zeros<Col<T> >(dim);

        int count_num = 0;
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            int oa = M_scheme[loop][i].index;
            int la = M_scheme[loop][i].l;
            int ta = (M_scheme[loop][i].Iz>=0)?1:-1;
            int ja = int(M_scheme[loop][i].j*2+0.5);
            int ma = int(M_scheme[loop][i].m*2+0.5+ja)-ja;
            for(int j=i-1;j>=0;j--)
            {
                int ob = M_scheme[loop][j].index;
                int lb = M_scheme[loop][j].l;
                int tb = (M_scheme[loop][j].Iz>=0)?1:-1;
                int jb = int(M_scheme[loop][j].j*2+0.5);
                int mb = int(M_scheme[loop][j].m*2+0.5+jb)-jb;

                const int offset1 = (oa>=ob)?(oa*(oa+1)/2+ob):(ob*(ob+1)/2+oa);

                for(int k=Dimension_M[loop]-1;k>=0;k--)
                {
                    int oc = M_scheme[loop][k].index;
                    int lc = M_scheme[loop][k].l;
                    int tc = (M_scheme[loop][k].Iz>=0)?1:-1;
                    int jc = int(M_scheme[loop][k].j*2+0.5);
                    int mc = int(M_scheme[loop][k].m*2+0.5+jc)-jc;

                    for(int l=k-1;l>=0;l--)
                    {
                        int od = M_scheme[loop][l].index;
                        int ld = M_scheme[loop][l].l;
                        int td = (M_scheme[loop][l].Iz>=0)?1:-1;
                        int jd = int(M_scheme[loop][l].j*2+0.5);
                        int md = int(M_scheme[loop][l].m*2+0.5+jd)-jd;

                        int tij = ta+tb;
                        int tkl = tc+td;
                        int mij = ma+mb;
                        int mkl = mc+md;
                        int pij = (la+lb)&1;
                        int pkl = (lc+ld)&1;

                        const int offset2 = (oc>=od)?(oc*(oc+1)/2+od):(od*(od+1)/2+oc);
                        const int indx = (offset1>=offset2)?(offset1*(offset1+1)/2+offset2):(offset2*(offset2+1)/2+offset1);

                        const int offset = Orbit[indx].jmax-Orbit[indx].jmin+1;

                        if((mij==mkl)&&(tij==tkl)&&(pij==pkl))
                        {
                            double fact = (((jb-ja+jd-jc)/2)&1)?-1.0:1.0;
                            if(oa==ob) fact*=sqrt(2.0);
                            if(oc==od) fact*=sqrt(2.0);

                            int fact1;
                            double sum = 0.0;
                            for(int jr=max(abs(mij/2),Orbit[indx].jmin);jr<=Orbit[indx].jmax;jr++)
                            {
                                const int jr2 = (jr<<1);

                                fact1 = 1.0;
                                if(oa<ob)
                                {
                                    int sign = ((((ja+jb)>>1)-jr)&1)?-1:1;
                                    fact1 *= -sign;
                                }
                                if(oc<od)
                                {
                                    int sign = ((((jc+jd)>>1)-jr)&1)?-1:1;
                                    fact1 *= -sign;
                                }

                                int indx2 = IsoSpin_Z*IsoSpin_Z*(jr-Orbit[indx].jmin+offset);

                                sum += double(jr2+1)*gsl_sf_coupling_3j(ja,jb,jr2,ma,mb,-mij)*gsl_sf_coupling_3j(jc,jd,jr2,mc,md,-mkl)*(Orbit[indx].v(indx2)+Orbit[indx].v(indx2+loop+IsoSpin_Z))*fact1;

                            }
                            sum *= fact;

                            if(fabs(sum)>EPS)
                            {
                                uvec col;
                                col<<i<<j<<k<<l<<endr;

                                index_duplicate.col(count_num) = col;
                                TBME_duplicate(count_num) = sum;

                                count_num++;
                            }
                        }
                    }
                }
            }
        }

        index[loop]=index_duplicate.cols(0,count_num-1);
        TBME[loop]=TBME_duplicate.subvec(0,count_num-1);
    }


    ///pn interaction
    const int pair_dim = Dimension_M[0]*Dimension_M[1];
    const int dim = pair_dim*pair_dim;

    umat index_duplicate = zeros<umat>(FootnoteNum,dim);
    Col<T> TBME_duplicate = zeros<Col<T> >(dim);

    int count_num = 0;
    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        int oa = M_scheme[0][i].index;
        int la = M_scheme[0][i].l;
        int ta = (M_scheme[0][i].Iz>=0)?1:-1;
        int ja = int(M_scheme[0][i].j*2+0.5);
        int ma = int(M_scheme[0][i].m*2+0.5+ja)-ja;
        for(int j=Dimension_M[1]-1;j>=0;j--)
        {
            int ob = M_scheme[1][j].index;
            int lb = M_scheme[1][j].l;
            int tb = (M_scheme[1][j].Iz>=0)?1:-1;
            int jb = int(M_scheme[1][j].j*2+0.5);
            int mb = int(M_scheme[1][j].m*2+0.5+jb)-jb;

            const int offset1 = (oa>=ob)?(oa*(oa+1)/2+ob):(ob*(ob+1)/2+oa);

            for(int k=Dimension_M[0]-1;k>=0;k--)
            {
                int oc = M_scheme[0][k].index;
                int lc = M_scheme[0][k].l;
                int tc = (M_scheme[0][k].Iz>=0)?1:-1;
                int jc = int(M_scheme[0][k].j*2+0.5);
                int mc = int(M_scheme[0][k].m*2+0.5+jc)-jc;

                for(int l=Dimension_M[1]-1;l>=0;l--)
                {
                    int od = M_scheme[1][l].index;
                    int ld = M_scheme[1][l].l;
                    int td = (M_scheme[1][l].Iz>=0)?1:-1;
                    int jd = int(M_scheme[1][l].j*2+0.5);
                    int md = int(M_scheme[1][l].m*2+0.5+jd)-jd;

                    int tij = ta+tb;
                    int tkl = tc+td;
                    int mij = ma+mb;
                    int mkl = mc+md;
                    int pij = (la+lb)&1;
                    int pkl = (lc+ld)&1;

                    const int offset2 = (oc>=od)?(oc*(oc+1)/2+od):(od*(od+1)/2+oc);
                    const int indx = (offset1>=offset2)?(offset1*(offset1+1)/2+offset2):(offset2*(offset2+1)/2+offset1);

                    const int offset = Orbit[indx].jmax-Orbit[indx].jmin+1;

                    if((mij==mkl)&&(tij==tkl)&&(pij==pkl))
                    {
                        double fact = (((jb-ja+jd-jc)/2)&1)?-1.0:1.0;
                        if(oa==ob) fact*=sqrt(2.0);
                        if(oc==od) fact*=sqrt(2.0);

                        int fact0,fact1;
                        double sum = 0.0;
                        for(int jr=max(abs(mij/2),Orbit[indx].jmin);jr<=Orbit[indx].jmax;jr++)
                        {
                            const int jr2 = (jr<<1);

                            fact0 = 1.0;
                            fact1 = 1.0;
                            if(oa<ob)
                            {
                                int sign = ((((ja+jb)>>1)-jr)&1)?-1:1;
                                fact0 *= sign;
                                fact1 *= -sign;
                            }
                            if(oc<od)
                            {
                                int sign = ((((jc+jd)>>1)-jr)&1)?-1:1;
                                fact0 *= sign;
                                fact1 *= -sign;
                            }

                            int indx2 = IsoSpin_Z*IsoSpin_Z*(jr-Orbit[indx].jmin+offset);
                            int indx3 = IsoSpin_Z*IsoSpin_Z*(jr-Orbit[indx].jmin);

                            double temp = double(jr2+1)*gsl_sf_coupling_3j(ja,jb,jr2,ma,mb,-mij)*gsl_sf_coupling_3j(jc,jd,jr2,mc,md,-mkl);
                            sum += temp*fact0*(Orbit[indx].v(indx3)+Orbit[indx].v(indx3+1))*gsl_sf_coupling_3j(1,1,0,ta,tb,-tij)*gsl_sf_coupling_3j(1,1,0,tc,td,-tkl);
                            sum += 3.0*temp*fact1*(Orbit[indx].v(indx2)+Orbit[indx].v(indx2+1))*gsl_sf_coupling_3j(1,1,2,ta,tb,-tij)*gsl_sf_coupling_3j(1,1,2,tc,td,-tkl);
                        }
                        sum*=fact;

                        if(fabs(sum)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;

                            index_duplicate.col(count_num) = col;
                            TBME_duplicate(count_num) = sum;

                            count_num++;
                        }
                    }
                }
            }
        }
    }

    index[IsoSpin_Z] = index_duplicate.cols(0,count_num-1);
    TBME[IsoSpin_Z] = TBME_duplicate.subvec(0,count_num-1);
}

template <typename T>
void Interaction<T>::Read_Interaction_Iso (string ME)
/** \brief Read OBME and TBME in isospin representation
 *
 * \param MartrixElements
 * \return OBME
 * \return TBME J scheme
 *
 */
{
    const double Scale1 = 1.0;
    const double Scale2 = pow(18.0/(Mass_Num+16),0.3);

    int size_dim;
    int ia,ib,ic,id,j,iso;
    double v;
    ifstream ifile(ME.c_str());

    ifile>>size_dim;

    OBME[0]=zeros<Mat<T> >(Dimension_M[0],Dimension_M[0]);
    double *obme_J = new double [Dimension_J[0]];
    for(int i=0;i<Dimension_J[0];i++)
    {
        ifile>>obme_J[i];
    }

    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        OBME[0](i,i)=obme_J[M_scheme[0][i].index];
    }
    OBME[0]*=Scale1;
    OBME[1]=OBME[0];
    delete [] obme_J;

    int phase;
    for(int i=size_dim-1;i>=0;i--)
    {
        ifile>>ia;
        ifile>>ib;
        ifile>>ic;
        ifile>>id;
        ifile>>j;
        ifile>>iso;
        ifile>>v;

        ia--;
        ib--;
        ic--;
        id--;

        phase = 1;
        if(ia<ib)
        {
            swap(ia,ib);
            phase *= ((int(J_scheme[0][ia].j+J_scheme[0][ib].j+0.5)-j-iso)&1)?-1:1;
        }
        if(ic<id)
        {
            swap(ic,id);
            phase *= ((int(J_scheme[0][ic].j+J_scheme[0][id].j+0.5)-j-iso)&1)?-1:1;
        }
        if((ia<ic)||((ia==ic)&&(ib<id)))
        {
            swap(ia,ic);
            swap(ib,id);
        }
        const int offset1 = ia*(ia+1)/2+ib;
        const int offset2 = ic*(ic+1)/2+id;
        const int indx = offset1*(offset1+1)/2+offset2;
        Orbit[indx].v(iso*(Orbit[indx].jmax-Orbit[indx].jmin+1)+j-Orbit[indx].jmin) += v*Scale2*phase;
    }
    ifile.close();
}

template <typename T>
void Interaction<T>::Read_Interaction_PN_QBox (string ME)
{
	const double Scale1 = 1.0 - 1.0/Mass_Num;
	const double Scale2 = pow(18.0/(Mass_Num+16),0.3);

//	cout << Scale1 << endl;
	int size_dim;
	int ia, ib, ic, id, J, iso, par;
	double v1, v2, v3, v4, v, H_com, r_ir_j, p_ip_j;
	string stemp = "aaaaa";
	ifstream ifile(ME.c_str());

	Mat<T> OBME_nofix[2], OBME_comfix[2];

/*	while(stemp.substr(0,3) != " a,")
	    getline(ifile, stemp);

	int counter = 0;
	OBME[0] = zeros<Mat<T> >(Dimension_M[0], Dimension_M[0]);
	OBME_nofix[0] = zeros<Mat<T> >(Dimension_M[0], Dimension_M[0]);
	OBME_comfix[0]= zeros<Mat<T> >(Dimension_M[0], Dimension_M[0]);
	while(counter < Dimension_J[0]+Dimension_J[1])
	{
		ifile >> ia;
		ifile >> ic;
		ifile >> v1;
		ifile >> v2;
		ifile >> v3;
//		cout << ia << "  " << ic << "  " << v3 << endl;
//		cin >> stemp;
		if(ia == ic) counter++;
		if(1) v = v3;			///v3 equals total sp energy
		for(int i=Dimension_M[0]-1; i>=0; i--)
		    for(int j=Dimension_M[0]-1; j>=0; j--)
			{
				if(M_scheme[0][i].m != M_scheme[0][j].m) continue;
//				cout << i << "  " << j << "  " << M_scheme[0][i].m << "  " << M_scheme[0][j].m << endl;
//				cin >> stemp;
				if(idx_J[0][M_scheme[0][i].index] == ia && idx_J[0][M_scheme[0][j].index] == ic)
				{
					OBME_nofix[0](i,j) = v*Scale1;
					OBME_nofix[0](j,i) = v*Scale1;
					OBME_comfix[0](i,j) = v*Scale1;
				    OBME_comfix[0](j,i) = v*Scale1;
				}
				if(ia == ic && idx_J[0][M_scheme[0][i].index] == ia && j == i)
				{
					OBME_comfix[0](i,i) += beta/Mass_Num*v*2.;
					OBME_comfix[0](i,i) -= 1.5*beta*hbar_omega/Mass_Num;			//minus 3/2 hbar_omega in H tot
				}
			}
	}
	OBME_nofix[1] = OBME_nofix[0];
	OBME_comfix[1]= OBME_comfix[0];

	OBME[0] = OBME_comfix[0];
	OBME[1] = OBME_comfix[1];*/

	for(int loop=IsoSpin_Z-1;loop>=0;loop--)
	{
		OBME[loop]=zeros<Mat<T> >(Dimension_M[loop],Dimension_M[loop]);
		for(int i=Dimension_M[loop]-1;i>=0;i--)
		{
			int na = M_scheme[loop][i].n;
			int la = M_scheme[loop][i].l;
			int ta = (M_scheme[loop][i].Iz>=0)?1:-1;
			int ja = int(M_scheme[loop][i].j*2+0.5);
			int ma = int(M_scheme[loop][i].m*2+0.5+ja)-ja;
			for(int j=Dimension_M[loop]-1;j>=0;j--)
			{
				int nb = M_scheme[loop][j].n;
				int lb = M_scheme[loop][j].l;
				int tb = (M_scheme[loop][j].Iz>=0)?1:-1;
				int jb = int(M_scheme[loop][j].j*2+0.5);
				int mb = int(M_scheme[loop][j].m*2+0.5+jb)-jb;

				if((la==lb)&&(ta==tb)&&(ja==jb)&&(ma==mb))
				{
					if (na==nb)
					{
						OBME[loop](i,j)=(na<<1)+la+1.5;
					}
					else if (abs(na-nb)==1)
					{
						int n = max(na,nb);
						OBME[loop](i,j)=sqrt(n*(n+la+0.5));
					}
				}
			}
			OBME[loop](i,i) *= 1. + 2. * beta / (double(Mass_Num) * (1.-1./double(Mass_Num)));
			OBME[loop](i,i) -= 3. * beta / (double(Mass_Num) * (1.-1./double(Mass_Num)));
		}
		OBME[loop] *= 0.5*hbar_omega*(1.0-1.0/double(Mass_Num));
	}
	cout << OBME[0](0,0) << endl;

	ofstream obme_fin("obme_ini.dat");
	for(int i=Dimension_M[0]-1;i>=0;i--)
	{
		for(int j=Dimension_M[0]-1;j>=0;j--)
		{
			obme_fin << OBME[0](i,j).real() << "  ";
		}
		obme_fin << endl;
	}
	obme_fin.close();
/*	ofstream obme_ini("obme_ini.dat");
	for(int i=Dimension_M[0]-1;i>=0;i--)
	{
		for(int j=Dimension_M[0]-1;j>=0;j--)
		{
			obme_ini << OBME_nofix[0](i,j).real() << "  ";
		}
		obme_ini << endl;
	}
	obme_ini.close();*/
	cout << OBME[0](0,0) << endl;

	cout << "OBME complete" << endl;

//	getline(ifile, stemp);
//	getline(ifile, stemp);
//	getline(ifile, stemp);
//	cout << stemp << endl;

	while(stemp!="Tz")
	{
		ifile >> stemp;
	}
	getline(ifile, stemp);

	ofstream vcompare("v_com.dat");
	int phase;
	while(!ifile.eof())
	{
		ifile >> iso;
		if(ifile.eof())break;
		ifile >> par;
		ifile >> J;
		J /= 2;
		ifile >> ia;
		ifile >> ib;
		ifile >> ic;
		ifile >> id;
		ifile >> v1;
		ifile >> v2;
		ifile >> v3;
		ifile >> v4;
		v = v1;
		H_com = v2;
		r_ir_j = v3;
		p_ip_j = v4;
//		if(fabs((v7-v6)/v6)>1) cout << "  " << ia << "  " << ib << "  " << ic << "  " << id << "  " << j << "  " << v6 << "  " << v7 << endl;

//		cout << ia << "  " << ib << "  " << ic << "  " << id << "  " << j << "  " << iso << "  " << v << endl;
//		cin >> stemp;
		int iax, ibx, icx, idx;
		for(int i=0; i<=1; i++)
		    for(int j=Dimension_J[0]-1; j>=0; j--)
			{
				if(idx_J[i][j] == ia) iax = j;
				if(idx_J[i][j] == ib) ibx = j;
				if(idx_J[i][j] == ic) icx = j;
				if(idx_J[i][j] == id) idx = j;
			}

//		cout << iax << "  " << ibx << "  " << icx << "  " << idx << endl;
//		cin >> stemp;

		phase = 1;
		if(iso != 0)
		{
			if(iax < ibx)
			{
				swap(iax, ibx);
				phase *= ((int(J_scheme[(iso+1)/2][iax].j + J_scheme[(iso+1)/2][ibx].j + 0.5) - J) & 1) ? 1 : -1;
			}
			if(icx < idx)
			{
				swap(icx, idx);
				phase *= ((int(J_scheme[(iso+1)/2][icx].j + J_scheme[(iso+1)/2][idx].j + 0.5) - J) & 1) ? 1 : -1;
			}
			if((iax < icx) || ((iax == icx) && (ibx < idx)))
			{
				swap(iax, icx);
				swap(ibx, idx);
			}
		}
		else
		{
			if((ia&0)||(ic&0))
			{
				phase *= (int(J_scheme[1][iax].j+J_scheme[0][ibx].j+J_scheme[1][icx].j+J_scheme[0][idx].j+0.5)&1)?-1:1;
				swap(iax,ibx);
				swap(icx,idx);
			}
			if(iax<icx||(iax==icx&&ibx<idx))
			{
				swap(iax,icx);
				swap(ibx,idx);
			}
		}
		int offset1 = iax * (iax + 1) / 2 + ibx;
		if(iso == 0) offset1 = iax * Dimension_J[1] + ibx;
		int offset2 = icx * (icx + 1) / 2 + idx;
		if(iso == 0) offset2 = icx * Dimension_J[1] + idx;
		const int dim = Dimension_J[0] * (Dimension_J[0]+1)/2;
		const int offset = dim * (dim + 1) / 2;
		const int indx = offset1 * (offset1 + 1) / 2 + offset2;
		int indx1;
//		cout << "hello" << endl;
//		cout.flush();
		if(iso == -1 || iso == 1) indx1 = indx + (iso+1)/2 * offset;
		if(iso == 0) indx1 = indx + 2*offset;
//		cout << indx1 << "  " << j << "  " << Orbit[indx1].jmin << "  " << Orbit[indx1].jmax << endl;
//		cout << iax << "  " << ibx << "  " << icx << "  " << idx << endl;
//		cout.flush();
		/*if(fabs(v-( v + beta/Mass_Num*hbar_omega*H_com - p_ip_j/Mass_Num*hbar_omega ))>0.0001)*/vcompare << ( v + beta/Mass_Num*hbar_omega*H_com - p_ip_j/double(Mass_Num)*hbar_omega) * phase << "  " << indx1 << "  " << J-Orbit[indx1].jmin << "  " << iso << endl;
		/*if(j >= Orbit[indx1].jmin && j <= Orbit[indx1].jmax)*/ Orbit[indx1].v(J-Orbit[indx1].jmin) += /*Scale2**/( v + beta/Mass_Num*hbar_omega*H_com - p_ip_j/double(Mass_Num)*hbar_omega ) * phase;
//		cout << iso << "  " << indx1 << endl;
//		cout.flush();
//		cin >> stemp;
//		cout << v /*-p_ip_j/double(Mass_Num)*hbar_omega*/ << "  " << phase <<endl;
//		cin >> stemp;
	}
	vcompare.close();
	ifile.close();
	cout << "Read Int complete" << endl;
//	cin >> stemp;
}

template <typename T>
void Interaction<T>::Read_Interaction_PN (string ME)
/** \brief Read OBME and TBME in proton-neutron representation
 *
 * \param MartrixElements
 * \return OBME
 * \return TBME J scheme
 *
 */
{
/*
    const double Scale1 = 1.0;
    const double Scale2 = pow(18.0/(Mass_Num+16),0.3);

    int dim;
    int ia,ib,ic,id,j,tz;
    double v;
    ifstream ifile(ME.c_str());

    ifile>>dim;
    OBME[0]=zeros<Mat<T> >(Dimension_M[0],Dimension_M[0]);
    double *obme_J = new double [Dimension_J[0]];
    for(int i=0;i<Dimension_J[0];i++)
    {
        ifile>>obme_J[i];
    }

    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        OBME[0](i,i)=obme_J[M_scheme[0][i].index];
    }
    OBME[0]*=Scale1;
    OBME[1]=OBME[0];
    delete [] obme_J;

    int phase;
    for(int i=dim-1;i>=0;i--)
    {
        ifile>>ia;
        ifile>>ib;
        ifile>>ic;
        ifile>>id;
        ifile>>j;
        ifile>>v;


        ia--;
        ib--;
        ic--;
        id--;

        tz = -1;
        if(ia>=Dimension_J[0]) tz++;
        if(ib>=Dimension_J[0]) tz++;

        if(tz!=0)
        {
            tz++;
            tz>>=1;

            int offset = Dimension_J[0]*(Dimension_J[0]+1)/2;
            offset = tz*offset*(offset+1)/2;

        ia=(ia>=Dimension_J[0])?(ia-Dimension_J[0]):ia;
        ib=(ib>=Dimension_J[0])?(ib-Dimension_J[0]):ib;
        ic=(ic>=Dimension_J[0])?(ic-Dimension_J[0]):ic;
        id=(id>=Dimension_J[0])?(id-Dimension_J[0]):id;


            phase = 1;
            if(ia<ib)
            {
                swap(ia,ib);
                phase *= ((int(J_scheme[tz][ia].j+J_scheme[tz][ib].j+0.5)-j)&1)?1:-1;
            }
            if(ic<id)
            {
                swap(ic,id);
                phase *= ((int(J_scheme[tz][ic].j+J_scheme[tz][id].j+0.5)-j)&1)?1:-1;
            }
            if((ia<ic)||((ia==ic)&&(ib<id)))
            {
                swap(ia,ic);
                swap(ib,id);
            }

            const int offset1 = ia*(ia+1)/2+ib;
            const int offset2 = ic*(ic+1)/2+id;
            const int indx = offset1*(offset1+1)/2+offset2+offset;

            Orbit[indx].v(j-Orbit[indx].jmin) += v*phase*Scale2;
        }
        else
        {
            int offset = 0;
            for(int k = IsoSpin_Z-1;k>=0;k--)
            {
                int temp = Dimension_J[k]*(Dimension_J[k]+1)/2;
                offset += temp*(temp+1)/2;
            }

            phase = 1;
            if((ia<Dimension_J[0])&&(ic<Dimension_J[0]))
            {
                phase *= (int(J_scheme[1][(ia)].j+J_scheme[0][(ib-Dimension_J[0])].j+J_scheme[1][(ic)].j+J_scheme[0][(id-Dimension_J[0])].j+0.5)&1)?-1:1;

                swap(ia,ib);
                swap(ic,id);
            }

        ia=(ia>=Dimension_J[0])?(ia-Dimension_J[0]):ia;
        ib=(ib>=Dimension_J[0])?(ib-Dimension_J[0]):ib;
        ic=(ic>=Dimension_J[0])?(ic-Dimension_J[0]):ic;
        id=(id>=Dimension_J[0])?(id-Dimension_J[0]):id;

            if((ia<ic)||((ia==ic)&&(ib<id)))
            {
                swap(ia,ic);
                swap(ib,id);
            }

            const int offset1 = ia*Dimension_J[1]+ib;
            const int offset2 = ic*Dimension_J[1]+id;
            const int indx = offset1*(offset1+1)/2+offset2+offset;

            Orbit[indx].v(j-Orbit[indx].jmin) += v*phase*Scale2;
        }
    }
    ifile.close();
*/


    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        OBME[loop]=zeros<Mat<T> >(Dimension_M[loop],Dimension_M[loop]);
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            int na = M_scheme[loop][i].n;
            int la = M_scheme[loop][i].l;
            int ta = (M_scheme[loop][i].Iz>=0)?1:-1;
            int ja = int(M_scheme[loop][i].j*2+0.5);
            int ma = int(M_scheme[loop][i].m*2+0.5+ja)-ja;
            for(int j=Dimension_M[loop]-1;j>=0;j--)
            {
                int nb = M_scheme[loop][j].n;
                int lb = M_scheme[loop][j].l;
                int tb = (M_scheme[loop][j].Iz>=0)?1:-1;
                int jb = int(M_scheme[loop][j].j*2+0.5);
                int mb = int(M_scheme[loop][j].m*2+0.5+jb)-jb;

                if((la==lb)&&(ta==tb)&&(ja==jb)&&(ma==mb))
                {
                    if (na==nb)
                    {
                        OBME[loop](i,j)=(na<<1)+la+1.5;
                    }
                    else if (abs(na-nb)==1)
                    {
                        int n = max(na,nb);
                        OBME[loop](i,j)=sqrt(n*(n+la+0.5));
                    }
                }
            }
        }
        OBME[loop] *= 0.5*hbar_omega*(1.0-1.0/double(Mass_Num));
    }


    int dim;
    ifstream ifile(ME.c_str());
    string s;

    do
    {
        ifile>>s;
    }
    while(s!="elements:");

    ifile>>dim;

    do
    {
        ifile>>s;
    }
    while(s!="<ab|p_ip_j|cd>");

    int phase;
    int ia,ib,ic,id,j,tz,p;
    double v,Hcom,rirj,pipj;
    for(int i=dim-1;i>=0;i--)
    {
        ifile>>tz;
        ifile>>p;
        ifile>>j;
        ifile>>ia;
        ifile>>ib;
        ifile>>ic;
        ifile>>id;
        ifile>>v;
        ifile>>Hcom;
        ifile>>rirj;
        ifile>>pipj;

        ia--;
        ib--;
        ic--;
        id--;

        j>>=1;

        if(tz!=0)
        {
            tz++;
            tz>>=1;

            int offset = Dimension_J[0]*(Dimension_J[0]+1)/2;
            offset = tz*offset*(offset+1)/2;

            ia>>=1;
            ib>>=1;
            ic>>=1;
            id>>=1;

            phase = 1;
            if(ia<ib)
            {
                swap(ia,ib);
                phase *= ((int(J_scheme[tz][ia].j+J_scheme[tz][ib].j+0.5)-j)&1)?1:-1;
            }
            if(ic<id)
            {
                swap(ic,id);
                phase *= ((int(J_scheme[tz][ic].j+J_scheme[tz][id].j+0.5)-j)&1)?1:-1;
            }
            if((ia<ic)||((ia==ic)&&(ib<id)))
            {
                swap(ia,ic);
                swap(ib,id);
            }

            const int offset1 = ia*(ia+1)/2+ib;
            const int offset2 = ic*(ic+1)/2+id;
            const int indx = offset1*(offset1+1)/2+offset2+offset;

            Orbit[indx].v(j-Orbit[indx].jmin) += (v-hbar_omega*pipj/double(Mass_Num))*phase;
        }
        else
        {
            int offset = 0;
            for(int k = IsoSpin_Z-1;k>=0;k--)
            {
                int temp = Dimension_J[k]*(Dimension_J[k]+1)/2;
                offset += temp*(temp+1)/2;
            }

            phase = 1;
            if((ia&1)&&(ic&1))
            {
                phase *= (int(J_scheme[1][(ia>>1)].j+J_scheme[0][(ib>>1)].j+J_scheme[1][(ic>>1)].j+J_scheme[0][(id>>1)].j+0.5)&1)?-1:1;

                swap(ia,ib);
                swap(ic,id);
            }

            ia>>=1;
            ib>>=1;
            ic>>=1;
            id>>=1;

            if((ia<ic)||((ia==ic)&&(ib<id)))
            {
                swap(ia,ic);
                swap(ib,id);
            }

            const int offset1 = ia*Dimension_J[1]+ib;
            const int offset2 = ic*Dimension_J[1]+id;
            const int indx = offset1*(offset1+1)/2+offset2+offset;

            Orbit[indx].v(j-Orbit[indx].jmin) += (v-hbar_omega*pipj/double(Mass_Num))*phase;
        }
    }
    ifile.close();
}

template <typename T>
void Interaction<T>::Read_Interaction_OLS (string ME)
/** \brief Read OBME and TBME in isospin representation
 *
 * \param MartrixElements
 * \return OBME
 * \return TBME J scheme
 *
 */
{
    double Scale1 = 2.0*hbar_omega/double(Mass_Num);
    double Scale2 = sqrt(938.92/938.093);
    int size_dim;
    int ia,ib,ic,id,j,iso;
    double Trelx,Hrelx,Vcoulx,Goutpn,Goutpp,Goutnn;
    ifstream ifile(ME.c_str());

    ifile>>size_dim;

    OBME[0]=zeros<Mat<T> >(Dimension_M[0],Dimension_M[0]);
    OBME[1]=zeros<Mat<T> >(Dimension_M[1],Dimension_M[1]);

    int phase;
    for(int i=size_dim-1;i>=0;i--)
    {
        ifile>>ia;
        ifile>>ib;
        ifile>>ic;
        ifile>>id;
        ifile>>j;
        ifile>>iso;
        ifile>>Trelx;
        ifile>>Hrelx;
        ifile>>Vcoulx;
        ifile>>Goutpn;
        ifile>>Goutpp;
        ifile>>Goutnn;
//Goutpn*=1.3;
//Goutpp*=1.3;
//Goutnn*=1.3;

        ia--;
        ib--;
        ic--;
        id--;

        phase = 1;
        if(ia<ib)
        {
            swap(ia,ib);
            phase *= ((int(J_scheme[0][ia].j+J_scheme[0][ib].j+0.5)-j-iso)&1)?-1:1;
        }
        if(ic<id)
        {
            swap(ic,id);
            phase *= ((int(J_scheme[0][ic].j+J_scheme[0][id].j+0.5)-j-iso)&1)?-1:1;
        }
        if((ia<ic)||((ia==ic)&&(ib<id)))
        {
            swap(ia,ic);
            swap(ib,id);
        }
        const int offset1 = ia*(ia+1)/2+ib;
        const int offset2 = ic*(ic+1)/2+id;
        const int indx = offset1*(offset1+1)/2+offset2;
        const int indx2 = IsoSpin_Z*IsoSpin_Z*(iso*(Orbit[indx].jmax-Orbit[indx].jmin+1)+j-Orbit[indx].jmin);
        Orbit[indx].v(indx2) += Trelx*Scale1*phase;
        Orbit[indx].v(indx2+1) += (Goutpn)*phase;
        if(iso==1)
        {
            Orbit[indx].v(indx2+2) += (Goutpp+Vcoulx*Scale2)*phase;
            Orbit[indx].v(indx2+3) += (Goutnn)*phase;
        }
    }
    ifile.close();
}

template <typename T>
void Interaction<T>::unpack (size_t num, int &i, int &j, int &k, int &l)
{
    const int count_num = 4;
    const int offset = sizeof(num)*8/count_num;
    const size_t temp = (size_t(1)<<offset)-1;

    l = num&temp;
    num>>=offset;
    k = num&temp;
    num>>=offset;
    j = num&temp;
    num>>=offset;
    i = num&temp;
}

template <typename T>
void Interaction<T>::pack (int i, int j, int k, int l, size_t &num)
{
    const int count_num = 4;
    const int offset = sizeof(num)*8/count_num;

    num = (((((size_t(i)<<offset)|size_t(j))<<offset)|size_t(k))<<offset)|size_t(l);
}

template <typename T>
void Interaction<T>::unpack (size_t num, int &i, int &j)
{
    const int count_num = 2;
    const int offset = sizeof(num)*8/count_num;
    const size_t temp = (size_t(1)<<offset)-1;

    j = num&temp;
    num>>=offset;
    i = num&temp;
}

template <typename T>
void Interaction<T>::pack (int i, int j, size_t &num)
{
    const int count_num = 2;
    const int offset = sizeof(num)*8/count_num;

    num = (size_t(i)<<offset)|size_t(j);
}

template<typename T>
void Interaction<T>::G_Matrix (Interaction<T> & src, vec * e_sp, vec & e_Fermi, bool axial)
{
    const double EPS = 1e-4;

    vec eps_bar = zeros<vec>(IsoSpin_Z);
    Mat<T> B[IsoSpin_Z+1];
    Mat<T> A[IsoSpin_Z+1];
    vec * m_record = NULL;

    if(axial) m_record = new vec [IsoSpin_Z+1];

    src.TBME_MatrixForm_AS(B,m_record);
    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        uvec e_indx = find(e_sp[loop]<=e_Fermi(loop));
        eps_bar(loop) = e_indx.is_empty()?0:mean(e_sp[loop](e_indx));

        A[loop].eye(B[loop].n_rows,B[loop].n_cols);
        #pragma omp parallel for
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            if(e_sp[loop](i)<=e_Fermi(loop)) continue;

            const int offseti = i*(i-1)/2;
            double eps_i = e_sp[loop](i);

            for(int j=i-1;j>=0;j--)
            {
                if(e_sp[loop](j)<=e_Fermi(loop)) continue;
                double eps_j = e_sp[loop](j);

                for(int k=Dimension_M[loop]-1;k>=0;k--)
                {
                    const int offsetk = k*(k-1)/2;
                    double eps_k = e_sp[loop](k)>e_Fermi(loop)?(2*eps_bar(loop)-e_sp[loop](k)):e_sp[loop](k);

                    for(int l=k-1;l>=0;l--)
                    {
                        double eps_l = e_sp[loop](l)>e_Fermi(loop)?(2*eps_bar(loop)-e_sp[loop](l)):e_sp[loop](l);
                        double omega = 1.0/(eps_i+eps_j-eps_k-eps_l);

                        A[loop](offsetk+l,offseti+j) += B[loop](offsetk+l,offseti+j)*omega;
                    }
                }
            }
        }

        Mat<T> temp;
        if(m_record==NULL) temp = solve(A[loop],B[loop]);
        else
        {
            temp = zeros<Mat<T> >(B[loop].n_rows,B[loop].n_cols);
            int jmax = m_record[loop].is_empty()?-1:int(max(abs(m_record[loop]))+0.5);

            for(int i=-jmax;i<=jmax;i++)
            {
                uvec m_indx = find(abs(m_record[loop]-i)<0.5);

                temp(m_indx,m_indx) = solve(A[loop](m_indx,m_indx),B[loop](m_indx,m_indx));
            }
        }

        uvec no_zeros = find(abs(temp)>EPS);
        index[loop]=zeros<umat>(FootnoteNum,no_zeros.n_elem);
        TBME[loop]=zeros<Col<T> >(no_zeros.n_elem);
        int count_num = 0;
        for(int i=Dimension_M[loop]-1;i>=0;i--)
        {
            const int offseti = i*(i-1)/2;

            for(int j=i-1;j>=0;j--)
            {
                for(int k=Dimension_M[loop]-1;k>=0;k--)
                {
                    const int offsetk = k*(k-1)/2;
                    for(int l=k-1;l>=0;l--)
                    {
                        T value = temp(offseti+j,offsetk+l);
                       /// T value = temp(i*Dimension_M[loop]+j,k*Dimension_M[loop]+l);
                        if(abs(value)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;

                            index[loop].col(count_num)=col;
                            TBME[loop](count_num)=value;

                            count_num++;
                        }
                    }
                }
            }
        }
    }


    A[IsoSpin_Z].eye(B[IsoSpin_Z].n_rows,B[IsoSpin_Z].n_cols);
    #pragma omp parallel for
    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        if(e_sp[0](i)<=e_Fermi(0)) continue;

        const int offseti = i*Dimension_M[1];
        double eps_i = e_sp[0](i);

        for(int j=Dimension_M[1]-1;j>=0;j--)
        {
            if(e_sp[1](j)<=e_Fermi(1)) continue;
            double eps_j = e_sp[1](j);

            for(int k=Dimension_M[0]-1;k>=0;k--)
            {
                const int offsetk = k*Dimension_M[1];
                double eps_k = e_sp[0](k)>e_Fermi(0)?(2*eps_bar(0)-e_sp[0](k)):e_sp[0](k);

                for(int l=Dimension_M[1]-1;l>=0;l--)
                {
                    double eps_l = e_sp[1](l)>e_Fermi(1)?(2*eps_bar(1)-e_sp[1](l)):e_sp[1](l);
                    double omega = 1.0/(eps_i+eps_j-eps_k-eps_l);

                    A[IsoSpin_Z](offsetk+l,offseti+j) += B[IsoSpin_Z](offsetk+l,offseti+j)*omega;
                }
            }
        }
    }

    Mat<T> temp;
    if(m_record==NULL) temp = solve(A[IsoSpin_Z],B[IsoSpin_Z]);
    else
    {
        temp = zeros<Mat<T> >(B[IsoSpin_Z].n_rows,B[IsoSpin_Z].n_cols);
        int jmax = m_record[IsoSpin_Z].is_empty()?-1:int(max(abs(m_record[IsoSpin_Z]))+0.5);

        for(int i=-jmax;i<=jmax;i++)
        {
            uvec m_indx = find(abs(m_record[IsoSpin_Z]-i)<0.5);

            temp(m_indx,m_indx) = solve(A[IsoSpin_Z](m_indx,m_indx),B[IsoSpin_Z](m_indx,m_indx));
        }
    }

    uvec no_zeros = find(abs(temp)>EPS);
    index[IsoSpin_Z]=zeros<umat>(FootnoteNum,no_zeros.n_elem);
    TBME[IsoSpin_Z]=zeros<Col<T> >(no_zeros.n_elem);
    int count_num = 0;
    for(int i=Dimension_M[0]-1;i>=0;i--)
    {
        const int offseti = i*Dimension_M[1];
        for(int j=Dimension_M[1]-1;j>=0;j--)
        {
            for(int k=Dimension_M[0]-1;k>=0;k--)
            {
                const int offsetk = k*Dimension_M[1];
                for(int l=Dimension_M[1]-1;l>=0;l--)
                {
                    T value = temp(offseti+j,offsetk+l);
                    if(abs(value)>EPS)
                    {
                        uvec col;
                        col<<i<<j<<k<<l<<endr;

                        index[IsoSpin_Z].col(count_num)=col;
                        TBME[IsoSpin_Z](count_num)=value;

                        count_num++;
                    }
                }
            }
        }
    }

    if(m_record != NULL) delete [] m_record;
}

template<typename T>
void Interaction<T>::G_Matrix_MemOptimize (Interaction<T> & src, vec * e_sp, vec & e_Fermi, bool axial)
{
    const double EPS = 1e-4;

    vec eps_bar = zeros<vec>(IsoSpin_Z);
    SpMat<T> B[IsoSpin_Z+1];
    imat indx[IsoSpin_Z+1];
    imat m_rec[IsoSpin_Z+1];

    if(axial)
    {
        src.sort_index_M(indx,m_rec);
        src.TBME_MatrixForm_AS(B,indx);
    }
    else src.TBME_MatrixForm_AS(B,NULL);

    for(int loop=IsoSpin_Z-1;loop>=0;loop--)
    {
        uvec e_indx = find(e_sp[loop]<=e_Fermi(loop));
        eps_bar(loop) = e_indx.is_empty()?0:mean(e_sp[loop](e_indx));

        if(!axial)
        {
            Mat<T> B_bk(B[loop]);
            Mat<T> A;
            A.eye(B_bk.n_rows,B_bk.n_cols);

            #pragma omp parallel for
            for(int i=Dimension_M[loop]-1;i>=0;i--)
            {
                if(e_sp[loop](i)<=e_Fermi(loop)) continue;

                const int offseti = i*(i-1)/2;
                double eps_i = e_sp[loop](i);

                for(int j=i-1;j>=0;j--)
                {
                    if(e_sp[loop](j)<=e_Fermi(loop)) continue;
                    double eps_j = e_sp[loop](j);

                    for(int k=Dimension_M[loop]-1;k>=0;k--)
                    {
                        const int offsetk = k*(k-1)/2;
                        double eps_k = e_sp[loop](k)>e_Fermi(loop)?(2*eps_bar(loop)-e_sp[loop](k)):e_sp[loop](k);

                        for(int l=k-1;l>=0;l--)
                        {
                            double eps_l = e_sp[loop](l)>e_Fermi(loop)?(2*eps_bar(loop)-e_sp[loop](l)):e_sp[loop](l);
                            double omega = 1.0/(eps_i+eps_j-eps_k-eps_l);

                            A(offsetk+l,offseti+j) += B_bk(offsetk+l,offseti+j)*omega;
                        }
                    }
                }
            }

            Mat<T> temp = solve(A,B_bk);

            uvec no_zeros = find(abs(temp)>EPS);
            index[loop]=zeros<umat>(FootnoteNum,no_zeros.n_elem);
            TBME[loop]=zeros<Col<T> >(no_zeros.n_elem);
            int count_num = 0;
            for(int i=Dimension_M[loop]-1;i>=0;i--)
            {
                const int offseti = i*(i-1)/2;

                for(int j=i-1;j>=0;j--)
                {
                    for(int k=Dimension_M[loop]-1;k>=0;k--)
                    {
                        const int offsetk = k*(k-1)/2;
                        for(int l=k-1;l>=0;l--)
                        {
                            T value = temp(offseti+j,offsetk+l);
                            if(abs(value)>EPS)
                            {
                                uvec col;
                                col<<i<<j<<k<<l<<endr;

                                index[loop].col(count_num)=col;
                                TBME[loop](count_num)=value;

                                count_num++;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            const int jdim = m_rec[loop].n_cols;
            int count_num = 0;
            int offset = 0;
            for(int i=0;i<jdim;i++)
            {
                const int dim_bk = m_rec[loop](1,i);

                Mat<T> B_bk(B[loop].submat(offset,offset,size(dim_bk,dim_bk)));

                Mat<T> A;
                A.eye(B_bk.n_rows,B_bk.n_cols);

                #pragma omp parallel for
                for(int j=dim_bk-1;j>=0;j--)
                {
                    const int offsetj = j+offset;
                    int ii = indx[loop](0,offsetj);
                    int jj = indx[loop](1,offsetj);
                    double eps_i = e_sp[loop](ii);
                    double eps_j = e_sp[loop](jj);
                    if((eps_i<=e_Fermi(loop))||(eps_j<=e_Fermi(loop))) continue;

                    for(int k=dim_bk-1;k>=0;k--)
                    {
                        const int offsetk = k+offset;
                        int kk = indx[loop](0,offsetk);
                        int ll = indx[loop](1,offsetk);
                        double eps_k = e_sp[loop](kk)>e_Fermi(loop)?(2*eps_bar(loop)-e_sp[loop](kk)):e_sp[loop](kk);
                        double eps_l = e_sp[loop](ll)>e_Fermi(loop)?(2*eps_bar(loop)-e_sp[loop](ll)):e_sp[loop](ll);

                        double omega = 1.0/(eps_i+eps_j-eps_k-eps_l);
                        A(k,j) += B_bk(k,j)*omega;
                    }
                }

                Mat<T> temp = solve(A,B_bk);

                uvec no_zeros = find(abs(temp)>EPS);
                index[loop].resize(FootnoteNum,count_num+no_zeros.n_elem);
                TBME[loop].resize(count_num+no_zeros.n_elem);

                for(int j=dim_bk-1;j>=0;j--)
                {
                    const int offsetj = j+offset;
                    int ii = indx[loop](0,offsetj);
                    int jj = indx[loop](1,offsetj);
                    for(int k=dim_bk-1;k>=0;k--)
                    {
                        T value = temp(j,k);
                        if(abs(value)>EPS)
                        {
                            const int offsetk = k+offset;
                            int kk = indx[loop](0,offsetk);
                            int ll = indx[loop](1,offsetk);

                            uvec col;
                            col<<ii<<jj<<kk<<ll<<endr;

                            index[loop].col(count_num)=col;
                            TBME[loop](count_num)=value;

                            count_num++;
                        }
                    }
                }
                offset += dim_bk;
            }
        }
    }

    if(!axial)
    {
        Mat<T> B_bk(B[IsoSpin_Z]);
        Mat<T> A;
        A.eye(B_bk.n_rows,B_bk.n_cols);

        #pragma omp parallel for
        for(int i=Dimension_M[0]-1;i>=0;i--)
        {
            if(e_sp[0](i)<=e_Fermi(0)) continue;

            const int offseti = i*Dimension_M[1];
            double eps_i = e_sp[0](i);

            for(int j=Dimension_M[1]-1;j>=0;j--)
            {
                if(e_sp[1](j)<=e_Fermi(1)) continue;
                double eps_j = e_sp[1](j);

                for(int k=Dimension_M[0]-1;k>=0;k--)
                {
                    const int offsetk = k*Dimension_M[1];
                    double eps_k = e_sp[0](k)>e_Fermi(0)?(2*eps_bar(0)-e_sp[0](k)):e_sp[0](k);

                    for(int l=Dimension_M[1]-1;l>=0;l--)
                    {
                        double eps_l = e_sp[1](l)>e_Fermi(1)?(2*eps_bar(1)-e_sp[1](l)):e_sp[1](l);
                        double omega = 1.0/(eps_i+eps_j-eps_k-eps_l);

                        A(offsetk+l,offseti+j) += B_bk(offsetk+l,offseti+j)*omega;
                    }
                }
            }
        }

        Mat<T> temp = solve(A,B_bk);

        uvec no_zeros = find(abs(temp)>EPS);
        index[IsoSpin_Z]=zeros<umat>(FootnoteNum,no_zeros.n_elem);
        TBME[IsoSpin_Z]=zeros<Col<T> >(no_zeros.n_elem);
        int count_num = 0;
        for(int i=Dimension_M[0]-1;i>=0;i--)
        {
            const int offseti = i*Dimension_M[1];
            for(int j=Dimension_M[1]-1;j>=0;j--)
            {
                for(int k=Dimension_M[0]-1;k>=0;k--)
                {
                    const int offsetk = k*Dimension_M[1];
                    for(int l=Dimension_M[1]-1;l>=0;l--)
                    {
                        T value = temp(offseti+j,offsetk+l);
                        if(abs(value)>EPS)
                        {
                            uvec col;
                            col<<i<<j<<k<<l<<endr;

                            index[IsoSpin_Z].col(count_num)=col;
                            TBME[IsoSpin_Z](count_num)=value;

                            count_num++;
                        }
                    }
                }
            }
        }
    }
    else
    {
        const int jdim = m_rec[IsoSpin_Z].n_cols;
        int count_num = 0;
        int offset = 0;
        for(int i=0;i<jdim;i++)
        {
            const int dim_bk = m_rec[IsoSpin_Z](1,i);

            Mat<T> B_bk(B[IsoSpin_Z].submat(offset,offset,size(dim_bk,dim_bk)));

            Mat<T> A;
            A.eye(B_bk.n_rows,B_bk.n_cols);

            #pragma omp parallel for
            for(int j=dim_bk-1;j>=0;j--)
            {
                const int offsetj = j+offset;
                int ii = indx[IsoSpin_Z](0,offsetj);
                int jj = indx[IsoSpin_Z](1,offsetj);
                double eps_i = e_sp[0](ii);
                double eps_j = e_sp[1](jj);
                if((eps_i<=e_Fermi(0))||(eps_j<=e_Fermi(1))) continue;

                for(int k=dim_bk-1;k>=0;k--)
                {
                    const int offsetk = k+offset;
                    int kk = indx[IsoSpin_Z](0,offsetk);
                    int ll = indx[IsoSpin_Z](1,offsetk);
                    double eps_k = e_sp[0](kk)>e_Fermi(0)?(2*eps_bar(0)-e_sp[0](kk)):e_sp[0](kk);
                    double eps_l = e_sp[1](ll)>e_Fermi(1)?(2*eps_bar(1)-e_sp[1](ll)):e_sp[1](ll);

                    double omega = 1.0/(eps_i+eps_j-eps_k-eps_l);
                    A(k,j) += B_bk(k,j)*omega;
                }
            }

            Mat<T> temp = solve(A,B_bk);

            uvec no_zeros = find(abs(temp)>EPS);
            index[IsoSpin_Z].resize(FootnoteNum,count_num+no_zeros.n_elem);
            TBME[IsoSpin_Z].resize(count_num+no_zeros.n_elem);

            for(int j=dim_bk-1;j>=0;j--)
            {
                const int offsetj = j+offset;
                int ii = indx[IsoSpin_Z](0,offsetj);
                int jj = indx[IsoSpin_Z](1,offsetj);
                for(int k=dim_bk-1;k>=0;k--)
                {
                    T value = temp(j,k);
                    if(abs(value)>EPS)
                    {
                        const int offsetk = k+offset;
                        int kk = indx[IsoSpin_Z](0,offsetk);
                        int ll = indx[IsoSpin_Z](1,offsetk);
                        uvec col;
                        col<<ii<<jj<<kk<<ll<<endr;

                        index[IsoSpin_Z].col(count_num)=col;
                        TBME[IsoSpin_Z](count_num)=value;

                        count_num++;
                    }
                }
            }
            offset += dim_bk;
        }
    }
}

template <typename T>
void Interaction<T>::ME_Output_Mscheme()
{
	ofstream OBME_out("obme.dat");
	ofstream TBME_out("tbme.dat");
	for(int iso=0; iso<=IsoSpin_Z-1; iso++)
	  for(int i=0; i<=OBME[iso].n_cols-1; i++)
		for(int j=i; j<=OBME[iso].n_cols-1; j++)
		{
			if(abs(OBME[iso](i,j))<=1e-4)continue;
			OBME_out << iso << "  " << i << "  " << j << "  " << OBME[iso](i,j) << endl;
		}
	OBME_out.close();
	TBME_out << "    Iso    a    b    c    d    <ab|V|cd>" << endl;
	for(int iso=0; iso<=Interaction<double>::IsoSpin_Z; iso++)
	{
		int Tz;
		if(iso == 0) Tz = -1;
		else if(iso == 1) Tz = 1;
		else if(iso == 2) Tz = 0;
		else Tz = 999;
		for(int count=0; count<=index[iso].n_cols-1; count++)
		{
			TBME_out << Tz << "  " << index[iso](0,count) << "  " << index[iso](1,count) << "  " << index[iso](2,count) << "  " << index[iso](3,count) << "  " << TBME[iso](count) << endl;
		}
	}
	TBME_out.close();
}

/***********************************************************************************************************************/


class HartreeFock
{
public:
    HartreeFock (int Z, int A, string space, string force, double bet);
    HartreeFock (int Z, int A, unsigned int * size_dim);
    HartreeFock (const HartreeFock & HF);
    ~HartreeFock () {}

    void operator = (const HartreeFock & HF);

    template<typename T1, typename T2>
    cx_double Operator_M (T1 & psil, T2 & psir, M_State * mscheme);
    template<typename T1, typename T2>
    cx_double Operator_P (T1 & psil, T2 & psir, M_State * mscheme);
    template<typename T1, typename T2>
    inline cx_mat Overlap_WaveFuction (T1 & psil, T2 & psir);
    template<typename T1, typename T2, typename T3>
    void Density_Matrix (T1 & psil, T2 & psir, T3 & over_matrix, cx_mat & rhoij, cx_double &overlap);
    template<typename T1, typename T2>
    void Density_Matrix (T1 & psil, T2 & psir, cx_mat & rhoij, cx_double &overlap);
    template<typename T>
    void Mean_Field (Interaction<T> & Force, cx_mat * rhoij, cx_mat * gamma);
    template<typename T>
    void Hamiltonian_sp (T & e_sp, cx_mat & gamma, cx_mat & ham);
    template<typename T1,typename T2>
    void Hamiltonian_sp (T1 & e_sp, cx_mat & gamma, cx_mat & ham, double q, double q0, Interaction<T2> & Force);
    template<typename T>
    cx_double Hamiltonian (Interaction<T> & Force, cx_mat * rhoij, cx_double * overlap);

    template<typename T>
    void Diagonalization (Interaction<T> & Force, double q20c=0.);
    template<typename T>
    void Initialization (Interaction<T> & Force);
    template<typename T>
    void Iteration (Interaction<T> & Force, double q20c=0);
    template<typename T>
    void Iteration_BHF (Interaction<T> & Force);
    void Broyden (cx_mat * m);

    template<typename T1, typename T2>
    void Rotation (double alpha, double beta, double gamma, T1 & src, T2 & dest, J_Orbit * jscheme, M_State * mscheme, int dim);
    void Space_Inversion (cx_mat & src, cx_mat & dest, M_State * mscheme);
    void AMP (cx_mat * psil, cx_mat * psir, vec & J, ivec & Parity, cx_vec & H, cx_vec & N);
    template<typename T1, typename T2, typename T3, typename T4>
    void AMP (T1 * psil, T2 * psir, vec & J, ivec & Parity, cx_mat * H, cx_mat * N, T3 * config, T4 & partition);

    template<typename T1, typename T>
    T Construct_ME (Interaction<T> & Force, T1 * psi_l, T1 * psi_r);
    template<typename T1, typename T>
    T Construct_ME2 (Interaction<T> & Force, T1 * psi_l, T1 * psi_r);


    void HW_Equation_Hermitian (cx_mat & A, cx_mat & B, vec & val, cx_mat & wf);
    void DBP ();

    size_t pack (ivec & config);
    ivec unpack (size_t config);
    int Count_Particle (size_t config);

    ///for deformation constraint
    template<typename T>
    void Deformation(Interaction<T> & Force, cx_mat* rho, double &dbeta, double &dgamma);

    template<typename T>
    cx_mat Q20_operator(Interaction<T> & Force);

	HF_SPS *hf_basis[Interaction<double>::IsoSpin_Z-1];						//interface for shell model by WuQiang
	void Set_HF_Basis();
	void Save_HF_Basis(string filename);

private:
    friend class Nucleus;

    Interaction<cx_double> ME;
    Interaction<cx_double> ME_HF;
    int MassNumber[Interaction<double>::IsoSpin_Z+1];

    cx_mat rho[Interaction<double>::IsoSpin_Z];
    cx_mat bra[Interaction<double>::IsoSpin_Z];
    cx_mat ket[Interaction<double>::IsoSpin_Z];
    cx_mat V_mf[Interaction<double>::IsoSpin_Z];

    cx_mat H_sp[Interaction<double>::IsoSpin_Z];
    vec E_sp[Interaction<double>::IsoSpin_Z];
    double BE;

    static const bool Axial_Symmetry = true;
    static const bool Parity_Decoupling = true;
    static const bool Deformation_Constrained = false;

    vec M_record[Interaction<double>::IsoSpin_Z];
    vec P_record[Interaction<double>::IsoSpin_Z];

    ///for deformation constraint
    double lambda;
    cx_mat Basis_Q20;
};

HartreeFock::HartreeFock (int Z, int A, string space, string force, double bet):ME(space,force,A,bet)
{
    MassNumber[0] = Z;
    MassNumber[1] = A-Z;
    MassNumber[2] = A;

    for(int i=Interaction<double>::IsoSpin_Z-1;i>=0;i--)
    {
        int size_dim = ME.Dimension_M[i];

        V_mf[i] = zeros<cx_mat>(size_dim,size_dim);
        if(Axial_Symmetry) M_record[i] = zeros<vec>(size_dim);
        if(Parity_Decoupling) P_record[i] = zeros<vec>(size_dim);
    }
}

HartreeFock::HartreeFock (int Z, int A, unsigned int * size_dim)
{
    MassNumber[0] = Z;
    MassNumber[1] = A-Z;
    MassNumber[2] = A;

    for(int i=Interaction<double>::IsoSpin_Z-1;i>=0;i--)
    {
        V_mf[i] = zeros<cx_mat>(size_dim[i],size_dim[i]);
        if(Axial_Symmetry) M_record[i] = zeros<vec>(size_dim[i]);
        if(Parity_Decoupling) P_record[i] = zeros<vec>(size_dim[i]);
    }
}

HartreeFock::HartreeFock (const HartreeFock & HF)
{
    (*this) = HF;
}

void HartreeFock::operator = (const HartreeFock & HF)
{
    const int iso = Interaction<double>::IsoSpin_Z;
    ME = HF.ME;
    ME_HF = HF.ME_HF;

    for(int i=iso;i>=0;i--) MassNumber[i] = HF.MassNumber[i];

    for(int i=iso-1;i>=0;i--)
    {
        rho[i] = HF.rho[i];
        bra[i] = HF.bra[i];
        ket[i] = HF.ket[i];
        V_mf[i] = HF.V_mf[i];

        H_sp[i] = HF.H_sp[i];
        E_sp[i] = HF.E_sp[i];

        M_record[i] = HF.M_record[i];
        P_record[i] = HF.P_record[i];
    }

    BE = HF.BE;
    lambda = HF.lambda;

    Basis_Q20 = HF.Basis_Q20;
}

template<typename T1, typename T2>
cx_double HartreeFock::Operator_M (T1 & psil, T2 & psir, M_State * mscheme)
/** \brief Subroutine to compute M quantum number of psi.
 *
 * \param psir right SD
 * \param psil left SD
 * \param Force Interaction
 * \return < psil | M | psir >
 *
 */
{
    cx_vec v = psil;
    for(int i=v.n_elem-1;i>=0;i--) v(i) *= mscheme[i].m;
    return dot(v,psir);
}

template<typename T1, typename T2>
cx_double HartreeFock::Operator_P (T1 & psil, T2 & psir, M_State * mscheme)
/** \brief Subroutine to compute M quantum number of psi.
 *
 * \param psir right SD
 * \param psil left SD
 * \param Force Interaction
 * \return < psil | M | psir >
 *
 */
{
    cx_vec v = psil;
    for(int i=v.n_elem-1;i>=0;i--)
    {
        if(mscheme[i].l&1) v(i) = -v(i);
    }
    return dot(v,psir);
}

template<typename T1, typename T2>
inline cx_mat HartreeFock::Overlap_WaveFuction (T1 & psil, T2 & psir)
{
    return psil.st()*psir;
}

template<typename T1, typename T2, typename T3>
void HartreeFock::Density_Matrix (T1 & psil, T2 & psir, T3 & over_matrix, cx_mat & rhoij, cx_double &overlap)
/** \brief Subroutine to compute rho(i,j) for two SD's that are not orthogonal using a matrix representation.
 *
 * \param psir right SD
 * \param psil left SD
 * \return rhoij: density matrix < psil | a^\dagger_i a_j | psir >
 * \return overlap: < psil | psir >
 *
 */
{
/*
    const double EPS = 1e-8;
    cx_mat U,V;
    vec S;
    svd_econ(U,S,V,over_matrix);
    S.elem( find(S< EPS) ).fill(EPS);
    over_matrix = U*diagmat(S)*V.t();
*/





    overlap = det(over_matrix);

    rhoij = psil*((psir*pinv(over_matrix)).st());
    ///!!!rhoij = psil*((psir*over_matrix.i()).st());
}

template<typename T1, typename T2>
void HartreeFock::Density_Matrix (T1 & psil, T2 & psir, cx_mat & rhoij, cx_double &overlap)
/** \brief Subroutine to compute rho(i,j) for two SD's that are not orthogonal using a matrix representation.
 *
 * \param psir right SD
 * \param psil left SD
 * \return rhoij: density matrix < psil | a^\dagger_i a_j | psir >
 * \return overlap: < psil | psir >
 *
 */
{
	cx_mat over_matrix = zeros<cx_mat>(psil.n_cols,psir.n_cols);
	for(int i=0; i<=psil.n_cols-1; i++)
	  for(int j=0; j<=psir.n_cols-1; j++)
		for(int k=0; k<=psil.n_rows-1; k++)
		{
			over_matrix(i,j) += psil(k,i) * psir(k,j);
		}
//    cx_mat over_matrix = psil.st()*psir;

/*
    const double EPS = 1e-8;
    cx_mat U,V;
    vec S;
    svd_econ(U,S,V,over_matrix);
    S.elem( find(S< EPS) ).fill(EPS);
    over_matrix = U*diagmat(S)*V.t();
*/





    overlap = det(over_matrix);

	rhoij = zeros <cx_mat> (psil.n_rows,psir.n_rows);
	cx_mat inv_over = pinv(over_matrix);
	for(int i=0; i<=psil.n_rows-1; i++)
	  for(int j=0; j<=psir.n_rows-1; j++)
		for(int k=0; k<=psil.n_cols-1; k++)
		  for(int l=0; l<=psir.n_cols-1; l++)
		  {
			  rhoij(i,j) += psil(i,k)*psir(j,l)*inv_over(l,k);
		  }
//    rhoij = psil*((psir*pinv(over_matrix)).st());
    ///!!!rhoij = psil*((psir*over_matrix.i()).st());


    ///!!!!!!
    ///SVD may be more suitable
    /*
    const double EPS = 0.00001;

    cx_mat U,V;
    vec S;
    svd(U,S,V,over_matrix.st());

    for(int i=S.n_elem-1;i>=0;i--)
    {
        if(fabs(S(i))>EPS) S(i) = 1.0/S(i);
        V.col(i) *= S(i);
    }
    rhoij = psil*(V*(U.t()))*(psir.st());
/////

    cx_mat U,V;
    vec S;
    svd(U,S,V,over_matrix.st());

    for(int i=S.n_elem-1;i>=0;i--)
    {
        vec ss = S;
        ss(i) = 1.0;

        V.col(i) *= prod(ss);
    }
    (V*(U.t())).print("Adjugate Matrix");

    rhoij = psil*(V*(U.t()))*(psir.st());*/

}

size_t HartreeFock::pack (ivec & config)
{
    size_t record = 0;
    for(int i=config.n_elem-1;i>=0;i--) record |= (size_t(1)<<config(i));

    return record;
}

ivec HartreeFock::unpack (size_t config)
{
    const int dim = sizeof(config)*8;

    ivec temp(dim);

    size_t probe = 1;
    int count_num = 0;
    for(int i=0;i<dim;i++)
    {
        if(probe&config) temp(count_num++) = i;

        probe<<=1;
    }

    if(count_num==0) return zeros<ivec>(0);
    return temp.subvec(0,(--count_num));
}

int HartreeFock::Count_Particle (size_t config)
{
    int count =0;
    while(config)
    {
        count++;
        config&=(config-1);
    }
    return count;
}

template<typename T>
void HartreeFock::Mean_Field (Interaction<T> & Force, cx_mat * rhoij, cx_mat * gamma)
/** \brief Calculates mean-field matrix
 *
 * \param Index(i) list of (packed) nonzero indices, i = 1 to nmatpp to get indices IJKL of i'th matrix element do UNPACK(Index)
 * \param TBME(i) ith TBME associated with indices unpacked from Index(i)
 * \param rhopij/rhonij density matices for protons/neutrons
 * \return gammap/gamman: Gamma matrix for protons/neutrons
 *
 */
{
    const int iso = Interaction<T>::IsoSpin_Z;

    for(int m=iso-1;m>=0;m--)
    {
        cx_mat * temp = &gamma[m];

        temp->fill(0);
        for(int n=Force.index[m].n_cols-1;n>=0;n--)
        {
            subview_col<uword> col = Force.index[m].col(n);
            int i=col(0);
            int j=col(1);
            int k=col(2);
            int l=col(3);
			T d = Force.TBME[m](n);
//            T d = Force.TBME[m](n)/2.;

            (*temp)(l,i) -= rhoij[m](j,k)*d;
            (*temp)(k,j) -= rhoij[m](i,l)*d;
            (*temp)(k,i) += rhoij[m](j,l)*d;
            (*temp)(l,j) += rhoij[m](i,k)*d;

//			(*temp)(i,l) -= rhoij[m](k,j)*d;
//			(*temp)(j,k) -= rhoij[m](l,i)*d;
//			(*temp)(i,k) += rhoij[m](l,j)*d;
//			(*temp)(j,l) += rhoij[m](k,i)*d;
        }

        const int pn = 1-m;
        if(m==0)
        {
            for(int n=Force.index[iso].n_cols-1;n>=0;n--)
            {
                subview_col<uword> col = Force.index[iso].col(n);
                int i=col(0);
                int j=col(1);
                int k=col(2);
                int l=col(3);
                (*temp)(k,i) += rhoij[pn](j,l)*Force.TBME[iso](n);

//				(*temp)(i,k) += rhoij[pn](l,j)*Force.TBME[iso](n)/2.;
            }
        }
        else
        {
            for(int n=Force.index[iso].n_cols-1;n>=0;n--)
            {
                subview_col<uword> col = Force.index[iso].col(n);
                int i=col(0);
                int j=col(1);
                int k=col(2);
                int l=col(3);
                (*temp)(l,j) += rhoij[pn](i,k)*Force.TBME[iso](n);

//				(*temp)(j,l) += rhoij[pn](k,i)*Force.TBME[iso](n)/2.;
			}
        }
    }
}

template<typename T>
cx_double HartreeFock::Hamiltonian (Interaction<T> & Force, cx_mat * rhoij, cx_double * overlap)
/** \brief < psil | H | psir >
 *
 * \param e_sp(i,j) one-body matrix element between i,j m-states
 * \param Index(i) list of (packed) nonzero indices, i = 1 to nmatpp to get indices IJKL of i'th matrix element do UNPACK(Index)
 * \param TBME(i) ith TBME associated with indices unpacked from Index(i)
 * \return rhoij density matrix < psil | a^\dagger_i a_j | psir >
 * \return < psil | H | psir >
 *
 */
{
    const int iso = Interaction<double>::IsoSpin_Z;

    cx_double energy = 0;

    ///OBME
    for(int m=iso-1;m>=0;m--) energy += trace(Force.OBME[m]*(rhoij[m].st()));

    ///pp, nn interaction
    for(int m=iso-1;m>=0;m--)
    {
        cx_mat * temp = &rhoij[m];

        for(int n=Force.index[m].n_cols-1;n>=0;n--)
        {
            subview_col<uword> col = Force.index[m].col(n);
            int i=col(0);
            int j=col(1);
            int k=col(2);
            int l=col(3);

            energy += (((*temp)(i,k))*((*temp)(j,l))-((*temp)(i,l))*((*temp)(j,k)))*Force.TBME[m](n);
        }
    }

    ///pn interaction
    for(int n=Force.index[iso].n_cols-1;n>=0;n--)
    {
        subview_col<uword> col = Force.index[iso].col(n);
        int i=col(0);
        int j=col(1);
        int k=col(2);
        int l=col(3);

        energy += Force.TBME[iso](n)*rhoij[0](i,k)*rhoij[1](j,l);
    }

    return energy*overlap[0]*overlap[1];
}

template<typename T>
void HartreeFock::Hamiltonian_sp (T & e_sp, cx_mat & gamma, cx_mat & ham)
/** \brief Calculates matrix h=T+Gamma
 *
 * \param e_sp(i,j) one-body matrix element between i,j m-states
 * \param gamma Gamma matrix
 * \return Hamiltonian_sp
 *
 */
{
    ham = gamma + e_sp;
}

template<typename T1,typename T2>
void HartreeFock::Hamiltonian_sp (T1 & e_sp, cx_mat & gamma, cx_mat & ham, double q, double q0, Interaction<T2> & Force)
/** \brief Calculates matrix h=T+Gamma+c(q-q0_lambda)Q
 *
 * \param e_sp(i,j) one-body matrix element between i,j m-states
 * \param gamma Gamma matrix
 * \return Hamiltonian_sp
 *
 */
{
    const double scale = 0.01;
    ham = gamma + e_sp + Q20_operator(Force)*(2.*scale*(q-q0+lambda));
}

template<typename T>
void HartreeFock::Diagonalization (Interaction<T> & Force, double q20c)
/** \brief Diagonlization single particle Harmiltonian
 *
 * \param Hamiltonian_sp
 * \return Energies
 * \return Wavefunctions
 *
 */
{
    const int iso = Interaction<T>::IsoSpin_Z;
    const int parity = Interaction<T>::Parity_Domain;
    cx_double overlap[iso];

    for(int i=iso-1;i>=0;i--)
    {
        if(MassNumber[i]>0)
        {
            subview<cx_double> bra_v = bra[i].cols(0,MassNumber[i]-1);
            subview<cx_double> ket_v = ket[i].cols(0,MassNumber[i]-1);

            Density_Matrix(bra_v,ket_v,rho[i],overlap[i]);
        }
        else
        {
            rho[i] = zeros<cx_mat>(Force.Dimension_M[i],Force.Dimension_M[i]);
            overlap[i] = 1;
        }
    }

    Mean_Field(Force,rho,V_mf);
	/*for(int i=0; i<=Force.index[0].n_cols-1; i++)
	{
		if(abs(Force.TBME[0](i))>=100.)
		  cout << i << "  " << Force.index[0](0,i) << "  " << Force.index[0](1,i) << "  " << Force.index[0](2,i) << "  " << Force.index[0](3,i) << "  " << Force.TBME[0](i) << endl;
	}*/
	/*double rhosum=0.;
	for(int i=0; i<=rho[0].n_rows-1; i++)
	{
		rhosum += rho[0](i,i).real();
		for(int j=0; j<=rho[0].n_cols-1; j++)
		{
			if(abs(rho[0](i,j))>=10.1)
			cout << i << "  " << j << "  " << rho[0](i,j) << endl;
		}
	}
	cout << rhosum << "!!!!!!!!" << endl;*/
    Broyden(V_mf);

    double dbeta;
    if(Deformation_Constrained)
    {
        dbeta=real(compute_trace(Basis_Q20,rho[0]+rho[1]));
        lambda += (dbeta - q20c);
    }

    for(int i=iso-1;i>=0;i--)
    {
        if(MassNumber[i]>0)
        {
            if(Deformation_Constrained) Hamiltonian_sp(Force.OBME[i],V_mf[i],H_sp[i],dbeta,q20c,Force);
            else Hamiltonian_sp(Force.OBME[i],V_mf[i],H_sp[i]);

            if(Axial_Symmetry&&Parity_Decoupling)
            {
                cx_mat wf = zeros<cx_mat>(H_sp[i].n_rows,H_sp[i].n_cols);
                vec eigen = zeros<vec>(Force.Dimension_M[i]);

                vec J_indx = zeros<vec>(Force.Dimension_M[i]);
                uvec P_indx = zeros<uvec>(Force.Dimension_M[i]);
                for(int m=Force.Dimension_M[i]-1;m>=0;m--)
                {
                    J_indx(m) = Force.M_scheme[i][m].m;
                    P_indx(m) = Force.M_scheme[i][m].l&1;
                }

                uvec indx(Force.Dimension_M[i]);
                double jmax = max(J_indx);
                for(int m=int(2*jmax+0.5);m>=0;m--)
                {
                    double m_quantum = double(m)-jmax;
                    int count_num = 0;

                    for(int n=Force.Dimension_M[i]-1;n>=0;n--)
                    {
                        if(int(fabs(m_quantum-Force.M_scheme[i][n].m)+0.5)==0)
                        {
                            indx(count_num) = n;
                            count_num++;
                        }
                    }

                    cx_mat wf_v;
                    vec eigen_v;
                    uvec indx_sub = indx.subvec(0,count_num-1);

                    for(int n=parity-1;n>=0;n--)
                    {
                        uvec indx_jp = find(P_indx(indx_sub)==n);

                        if(!indx_jp.is_empty())
                        {
                            uvec temp = indx_sub(indx_jp);

                            eig_sym(eigen_v,wf_v,H_sp[i](temp,temp));
                            wf(temp,temp) = wf_v;
                            eigen(temp) = eigen_v;
                        }
                    }
                }
                uvec eigen_indx = sort_index(eigen,"ascend");
                E_sp[i] = eigen(eigen_indx);

				/*cout << E_sp[i] << endl;
				string stemp;
				cin >> stemp;*/

				bra[i] = wf.cols(eigen_indx);
            }
            else if(Parity_Decoupling)
            {
                cx_mat wf = zeros<cx_mat>(H_sp[i].n_rows,H_sp[i].n_cols);
                vec eigen = zeros<vec>(Force.Dimension_M[i]);

                uvec P_indx = zeros<uvec>(Force.Dimension_M[i]);
                for(int m=Force.Dimension_M[i]-1;m>=0;m--) P_indx(m) = Force.M_scheme[i][m].l&1;

                for(int m=parity-1;m>=0;m--)
                {
                    cx_mat wf_v;
                    vec eigen_v;
                    uvec indx_sub = find(P_indx==m);

                    eig_sym(eigen_v,wf_v,H_sp[i](indx_sub,indx_sub));
                    wf(indx_sub,indx_sub) = wf_v;
                    eigen(indx_sub) = eigen_v;
                }
                uvec eigen_indx = sort_index(eigen,"ascend");
                E_sp[i] = eigen(eigen_indx);
                bra[i] = wf.cols(eigen_indx);
            }
            else if(Axial_Symmetry)
            {
                cx_mat wf = zeros<cx_mat>(H_sp[i].n_rows,H_sp[i].n_cols);
                vec eigen = zeros<vec>(Force.Dimension_M[i]);

                vec J_indx = zeros<vec>(Force.Dimension_M[i]);
                for(int m=Force.Dimension_M[i]-1;m>=0;m--) J_indx(m) = Force.M_scheme[i][m].m;

                uvec indx(Force.Dimension_M[i]);
                double jmax = max(J_indx);
                for(int m=int(2*jmax+0.5);m>=0;m--)
                {
                    double m_quantum = double(m)-jmax;
                    int count_num = 0;

                    for(int n=Force.Dimension_M[i]-1;n>=0;n--)
                    {
                        if(int(fabs(m_quantum-Force.M_scheme[i][n].m)+0.5)==0)
                        {
                            indx(count_num) = n;
                            count_num++;
                        }
                    }

                    cx_mat wf_v;
                    vec eigen_v;
                    uvec indx_sub = indx.subvec(0,count_num-1);

                    eig_sym(eigen_v,wf_v,H_sp[i](indx_sub,indx_sub));
                    wf(indx_sub,indx_sub) = wf_v;
                    eigen(indx_sub) = eigen_v;
                }
                uvec eigen_indx = sort_index(eigen,"ascend");
                E_sp[i] = eigen(eigen_indx);
                bra[i] = wf.cols(eigen_indx);
            }
            else eig_sym(E_sp[i],bra[i],H_sp[i],"std");

            ket[i] = conj(bra[i]);
        }
    }

    BE = Hamiltonian(Force,rho,overlap).real();
}

template<typename T>
void HartreeFock::Initialization (Interaction<T> & Force)
/** \brief Initialize the HF wavefunction
 *
 * \return Bra_WF
 * \return Ket_WF
 *
 */
{
    arma_rng::set_seed_random();

    for(int i=Interaction<double>::IsoSpin_Z-1;i>=0;i--)
    {
        if(MassNumber[i]>0)
        {
            cx_mat temp = randu<cx_mat>(Force.Dimension_M[i],Force.Dimension_M[i]);

            if(Axial_Symmetry||Parity_Decoupling)
            {
                for(int m=Force.Dimension_M[i]-1;m>=0;m--)
                {
                    for(int n=Force.Dimension_M[i]-1;n>m;n--) temp(m,n) = conj(temp(n,m));
                    for(int n=m;n>=0;n--)
                    {
                        if((Axial_Symmetry&&(int(fabs(Force.M_scheme[i][m].m-Force.M_scheme[i][n].m)+0.5)!=0))||(Parity_Decoupling&&((Force.M_scheme[i][m].l-Force.M_scheme[i][n].l)&1))) temp(m,n) = cx_double(0);
                    }
                }
            }

            cx_vec E;
            eig_gen(E,bra[i],temp);

            ket[i] = conj(bra[i]);
        }
    }

    if(Deformation_Constrained) Basis_Q20=Q20_operator(Force);
}

template<typename T>
void HartreeFock::Iteration (Interaction<T> & Force, double q20c)
{
    const bool Read_WF = false;
    const int Max_Iter = 10000000;
    const int iso = Interaction<T>::IsoSpin_Z;
    const double EPS = 1e-6;
	double E  = 1e6;
    int count_num = 0;

    Initialization(Force);

    string name = "WaveFunction_HF";
    if(Read_WF)
    {
        for(int i=iso-1;i>=0;i--)
        {
            stringstream sstm;
            sstm << name << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
            string result = sstm.str();
            cx_mat temp;
            if(temp.load(result.c_str())&&(temp.n_cols==Force.Dimension_M[i]))
            {
                bra[i] = temp;
                ket[i] = conj(bra[i]);
            }
        }
    }

    cout<<"Hartree-Fock iteration begins."<<endl;

    while(fabs(E-BE)/min(fabs(E),fabs(BE))>EPS)
    {
        E = BE;
		cout << E << endl;
		cout.flush();
        Diagonalization(Force,q20c);
        if(((count_num%20)==0)&&Read_WF) cout<<"Iteration "<<count_num<<", The g.s. is "<<BE<<" MeV."<<endl;
        count_num++;

        if(count_num>Max_Iter) break;
    }

    if(count_num<=Max_Iter) cout<<"Hartree Fock converged after "<<count_num<<" times iteration."<<endl;
    cout<<"The binding energy is "<<BE<<" MeV"<<endl;

    if(Deformation_Constrained)
    {
        double dbeta,dgamma;
        Deformation(Force,rho,dbeta,dgamma);
        cout<<"Value of Q20 is "<<real(compute_trace(Basis_Q20,rho[0]+rho[1]))<<"."<<endl;
        cout <<"beta: " << dbeta << ", " << "gamma: " << dgamma <<"."<< endl;
    }

    if(Axial_Symmetry)
    {
        for(int i=iso-1;i>=0;i--)
        {
            for(int j=bra[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> b_v = bra[i].col(j);
                subview_col<cx_double> k_v = ket[i].col(j);
                M_record[i](j) = Operator_M(b_v,k_v,Force.M_scheme[i]).real();
            }
        }
    }

    if(Parity_Decoupling)
    {
        for(int i=iso-1;i>=0;i--)
        {
            for(int j=bra[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> b_v = bra[i].col(j);
                subview_col<cx_double> k_v = ket[i].col(j);
                P_record[i](j) = Operator_P(b_v,k_v,Force.M_scheme[i]).real();
            }
        }
    }

    for(int i=iso-1;i>=0;i--)
    {
        const unsigned int show_levels = 20+MassNumber[i];
        const int size_dim = min(E_sp[i].n_elem,show_levels);

        if(size_dim>0)
        {
            cout<<"Single Particle Levels "<<i<<endl;
            cout<<"index "<<"\t E_sp ";
            if(Axial_Symmetry) cout<<"\t M quantum NO.";
            if(Parity_Decoupling) cout<<"\t Parity";
            cout<<endl;
        }

        for(int j=0;j<size_dim;j++)
        {
            cout<<j<<"\t"<<fixed<<setprecision(3)<<E_sp[i](j);
            if(Axial_Symmetry) cout<<"\t"<<setw(10)<<right<<M_record[i](j);
            if(Parity_Decoupling) cout<<"\t"<<setw(10)<<right<<P_record[i](j);
            cout<<endl;
        }
    }

    if(Read_WF)
    {
        for(int i=iso-1;i>=0;i--)
        {
            stringstream sstm;
            sstm << name << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
            string result = sstm.str();
            bra[i].save(result.c_str());
        }
    }
}

template<typename T>
void HartreeFock::Iteration_BHF (Interaction<T> & Force)
{
    const bool Read_WF = true;
    const int iso = Interaction<T>::IsoSpin_Z;
    const double EPS = 1e-6;
    double E = 1e6;
    int count_num = 0;

    Initialization(Force);


    bool Flag = true;
    string name_BHF = "WaveFunction_BHF";
    string name_Esp = "E_sp_BHF";
    if(Read_WF)
    {
        for(int i=iso-1;i>=0;i--)
        {
            stringstream sstm_BHF;
            sstm_BHF << name_BHF << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
            string result = sstm_BHF.str();
            cx_mat temp;
            if(temp.load(result.c_str())&&(temp.n_cols==Force.Dimension_M[i]))
            {
                bra[i] = temp;
                ket[i] = conj(bra[i]);
            }
            else Flag = false;

            stringstream Esp_BHF;
            Esp_BHF << name_Esp << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
            string result2 = Esp_BHF.str();
            vec temp2;
            if(temp2.load(result2.c_str())) E_sp[i] = temp2;
            else Flag = false;
        }
    }
    else Flag = false;

    if(!Flag)
    {
        string name = "WaveFunction_HF";
        if(Read_WF)
        {
            for(int i=iso-1;i>=0;i--)
            {
                stringstream sstm;
                sstm << name << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
                string result = sstm.str();
                cx_mat temp;
                if(temp.load(result.c_str())&&(temp.n_cols==Force.Dimension_M[i]))
                {
                    bra[i] = temp;
                    ket[i] = conj(bra[i]);
                }
            }
        }

        cout<<"Hartree-Fock iteration begins."<<endl;
        while(fabs(E-BE)>EPS)
        {
            E = BE;
            Diagonalization(Force);


            if((count_num%20)==0) cout<<"Iteration "<<count_num<<", The g.s. is "<<BE<<" MeV."<<endl;

            count_num++;
        }
        cout<<"Hartree-Fock converged after "<<count_num<<" times iteration."<<endl;
        cout<<"The binding energy is "<<BE<<" MeV"<<endl;

        if(Axial_Symmetry)
        {
            for(int i=iso-1;i>=0;i--)
            {
                for(int j=bra[i].n_cols-1;j>=0;j--)
                {
                    subview_col<cx_double> b_v = bra[i].col(j);
                    subview_col<cx_double> k_v = ket[i].col(j);
                    M_record[i](j) = Operator_M(b_v,k_v,Force.M_scheme[i]).real();
                }
            }
        }

///!!!!!!!!!!!!!!!!!!!!!!!!G_Mat P_record is unfinished.
        if(Parity_Decoupling)
        {
            for(int i=iso-1;i>=0;i--)
            {
                for(int j=bra[i].n_cols-1;j>=0;j--)
                {
                    subview_col<cx_double> b_v = bra[i].col(j);
                    subview_col<cx_double> k_v = ket[i].col(j);
                    P_record[i](j) = Operator_P(b_v,k_v,Force.M_scheme[i]).real();
                }
            }
        }

        for(int i=iso-1;i>=0;i--)
        {
            const unsigned int show_levels = 10+MassNumber[i];
            const int size_dim = min(E_sp[i].n_elem,show_levels);
            if(size_dim>0)
            {
                cout<<"Single Particle Levels "<<i<<endl;
                cout<<"index "<<"\t E_sp ";
                if(Axial_Symmetry) cout<<"\t M quantum NO.";
                if(Parity_Decoupling) cout<<"\t Parity";
                cout<<endl;
            }

            for(int j=0;j<size_dim;j++)
            {
                cout<<j<<"\t"<<fixed<<setprecision(3)<<E_sp[i](j);
                if(Axial_Symmetry) cout<<"\t"<<setw(10)<<right<<M_record[i](j);
                if(Parity_Decoupling) cout<<"\t"<<setw(10)<<right<<P_record[i](j);
                cout<<endl;
            }
        }

        for(int i=iso-1;i>=0;i--)
        {
            stringstream sstm;
            sstm << name << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
            string result = sstm.str();
            bra[i].save(result.c_str());
        }
    }


    Interaction<cx_double> HF_Basis;
    Interaction<cx_double> G_Mat = Force;

    cx_mat bra_i[iso],ket_i[iso];
    vec E_Fermi(iso);

    count_num = 0;
    cout<<"Brueckner-Hartree-Fock iteration begins."<<endl;

    do
    {
        if(Axial_Symmetry)
        {
            for(int i=iso-1;i>=0;i--)
            {
                for(int j=bra[i].n_cols-1;j>=0;j--)
                {
                    subview_col<cx_double> b_v = bra[i].col(j);
                    subview_col<cx_double> k_v = ket[i].col(j);
                    M_record[i](j) = Operator_M(b_v,k_v,G_Mat.M_scheme[i]).real();
                }
            }
        }

        if(Parity_Decoupling)
        {
            for(int i=iso-1;i>=0;i--)
            {
                for(int j=bra[i].n_cols-1;j>=0;j--)
                {
                    subview_col<cx_double> b_v = bra[i].col(j);
                    subview_col<cx_double> k_v = ket[i].col(j);
                    P_record[i](j) = Operator_P(b_v,k_v,G_Mat.M_scheme[i]).real();
                }
            }
        }

        HF_Basis.Basis_Transform(Force,bra,ket,M_record);

        for(int i=iso-1;i>=0;i--)
        {
            bra_i[i] = bra[i].i();
            ket_i[i] = ket[i].i();

            if(MassNumber[i]>0) E_Fermi(i) = E_sp[i](MassNumber[i]-1);
        }

        HF_Basis.G_Matrix(HF_Basis,E_sp,E_Fermi,Axial_Symmetry);
        G_Mat.Basis_Transform(HF_Basis,bra_i,ket_i);

        E = BE;
        Diagonalization(G_Mat);

        if((count_num%20)==0) cout<<"Iteration "<<count_num<<", The g.s. is "<<BE<<" MeV."<<endl;
        count_num++;
    }
    while(fabs(E-BE)>EPS);

    cout<<"Brueckner-Hartree-Fock converged after "<<count_num<<" times iteration."<<endl;
    cout<<"The binding energy is "<<BE<<" MeV"<<endl;

    if(Axial_Symmetry)
    {
        for(int i=iso-1;i>=0;i--)
        {
            for(int j=bra[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> b_v = bra[i].col(j);
                subview_col<cx_double> k_v = ket[i].col(j);
                M_record[i](j) = Operator_M(b_v,k_v,G_Mat.M_scheme[i]).real();
            }
        }
    }

    if(Parity_Decoupling)
    {
        for(int i=iso-1;i>=0;i--)
        {
            for(int j=bra[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> b_v = bra[i].col(j);
                subview_col<cx_double> k_v = ket[i].col(j);
                P_record[i](j) = Operator_P(b_v,k_v,G_Mat.M_scheme[i]).real();
            }
        }
    }

    for(int i=iso-1;i>=0;i--)
    {
        const unsigned int show_levels = 10+MassNumber[i];
        const int size_dim = min(E_sp[i].n_elem,show_levels);
        if(size_dim>0)
        {
            cout<<"Single Particle Levels "<<i<<endl;
            cout<<"index "<<"\t E_sp ";
            if(Axial_Symmetry) cout<<"\t M quantum NO.";
            if(Parity_Decoupling) cout<<"\t Parity";
            cout<<endl;
        }

        for(int j=0;j<size_dim;j++)
        {
            cout<<j<<"\t"<<fixed<<setprecision(3)<<E_sp[i](j);
            if(Axial_Symmetry) cout<<"\t"<<setw(10)<<right<<M_record[i](j);
            if(Parity_Decoupling) cout<<"\t"<<setw(10)<<right<<P_record[i](j);
            cout<<endl;
        }
    }

    for(int i=iso-1;i>=0;i--)
    {
        stringstream sstm_BHF;
        sstm_BHF << name_BHF << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
        string result = sstm_BHF.str();
        bra[i].save(result.c_str());

        stringstream Esp_BHF;
        Esp_BHF << name_Esp << i<<"_"<<MassNumber[0]<<"_"<<MassNumber[1];
        string result2 = Esp_BHF.str();
        E_sp[i].save(result2.c_str());
    }

    Force = G_Mat;
}

void HartreeFock::Broyden (cx_mat * m)
{
    static bool flag = true;
    static cx_mat old[Interaction<double>::IsoSpin_Z];

    const double scale = 0.7;

    if(flag)
    {
        for(int i=Interaction<double>::IsoSpin_Z-1;i>=0;i--)
        {
            old[i] = m[i];
        }
        flag = false;
    }
    else
    {
        for(int i=Interaction<double>::IsoSpin_Z-1;i>=0;i--)
        {
            m[i] = old[i]*(1.0-scale)+m[i]*scale;
            old[i] = m[i];
        }
    }
}

template<typename T1, typename T2>
void HartreeFock::Rotation (double alpha, double beta, double gamma, T1 & src, T2 & dest, J_Orbit * jscheme, M_State * mscheme, int dim)
/** \brief Rotate a state
 *
 * \param Wavefunction
 * \return Rotated Wavefunction
 *
 */
{
    Wigner_Function Coef;

    int count_num = 0;
    for(int i=0;i<dim;i++)
    {
        const double j = jscheme[i].j;
        const int size_dim = int(j*2+1.5);
        cx_mat rotmat = zeros<cx_mat>(size_dim,size_dim);

        for(int k=size_dim-1;k>=0;k--)
        {
            double m = mscheme[count_num+k].m;
            for(int l=size_dim-1;l>=0;l--)
            {
                rotmat(k,l)=Coef.Wigner_D_Matrix(j,m,mscheme[count_num+l].m,alpha,beta,gamma);
            }
        }

        dest.rows(count_num,count_num+size_dim-1) = rotmat*src.rows(count_num,count_num+size_dim-1);

        count_num+=size_dim;
    }
}

void HartreeFock::Space_Inversion (cx_mat & src, cx_mat & dest, M_State * mscheme)
/** \brief Parity projection
 *
 * \param Wavefunction
 * \return Wavefunction with space inversed
 *
 */
{
    cx_mat m = zeros<cx_mat>(dest.n_rows,dest.n_cols);
    for(int i=src.n_rows-1;i>=0;i--)
    {
        if(!(mscheme[i].l&1)) m.row(i) = src.row(i);
        else m.row(i) -= src.row(i);
    }
    dest=m;
}

void HartreeFock::AMP (cx_mat * psil, cx_mat * psir, vec & J, ivec & Parity, cx_vec & H, cx_vec & N)
/** \brief Angular momentum projection
 *
 * \param Bra_WF
 * \param Ket_WF
 * \param J angular momentum
 * \param P parity
 * \return Hamiltonian
 *
 */
{
    static bool flag = true;
    static double * theta_pos;
    static double * theta_w;
    static double * phi_pos;
    static double * phi_w;

    const int iso = Interaction<double>::IsoSpin_Z;
    const int npts = 20;
    const int npts_phi = Axial_Symmetry?1:npts;
    const int totalnum = npts*npts_phi*npts_phi;
    const int nthreads = omp_get_num_procs();
    const double jmin = min(J);
    Wigner_Function Coef;

    cx_mat rho_rot[nthreads*iso];
    cx_mat psir_rot[nthreads*iso];
    cx_vec H_thread[nthreads];
    cx_vec N_thread[nthreads];

    if(flag)
    {
        theta_pos = new double [npts];
        theta_w = new double [npts];
        phi_pos = new double [npts_phi];
        phi_w = new double [npts_phi];

        gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc (npts);
        for(int i=npts-1;i>=0;i--)
        {
            gsl_integration_glfixed_point(-1,1,i,&theta_pos[i],&theta_w[i],t);
            theta_pos[i]=acos(theta_pos[i]);
        }
        gsl_integration_glfixed_table_free(t);

        for(int i=npts_phi-1;i>=0;i--)
        {
            phi_w[i]=M_PI*2.0/double(npts_phi);
            phi_pos[i]=i*phi_w[i];
        }

        flag=false;
    }

    for(int i=nthreads-1;i>=0;i--)
    {
        const int offset = i*iso;
        for(int j=iso-1;j>=0;j--)
        {
            if(MassNumber[j]>0) psir_rot[offset+j] = zeros<cx_mat>(psir[j].n_rows,MassNumber[j]);
        }
        H_thread[i] = zeros<cx_vec>(H.n_elem);
        N_thread[i] = zeros<cx_vec>(N.n_elem);
    }

    omp_set_num_threads(nthreads);

    int psil_p = 1;
    int psir_p = 1;
    if(Parity_Decoupling)
    {
        for(int i=iso-1;i>=0;i--)
        {
            for(int j=psil[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> v = psil[i].col(j);
                psil_p *= int(Operator_P(v,conj(v),ME.M_scheme[i]).real()+1.5)-1;
            }

            for(int j=psir[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> v = psir[i].col(j);
                psir_p *= int(Operator_P(v,conj(v),ME.M_scheme[i]).real()+1.5)-1;
            }

            if(psil_p!=psir_p) return;
        }
    }

    #pragma omp parallel for
    for(int loop=totalnum-1;loop>=0;loop--)
    {
        int be = loop/npts_phi/npts_phi;
        int al = (loop/npts_phi)%npts_phi;
        int ga = loop%npts_phi;

        const int thread_id = omp_get_thread_num();
        const int offset = thread_id*iso;

        cx_mat * rho_pointer = &rho_rot[offset];
        cx_double overlap[iso];
        for(int i=iso-1;i>=0;i--)
        {
            if(MassNumber[i]>0)
            {
                subview<cx_double> psil_v = psil[i].cols(0,MassNumber[i]-1);
                subview<cx_double> psir_v = psir[i].cols(0,MassNumber[i]-1);
                Rotation(phi_pos[al],theta_pos[be],phi_pos[ga],psir_v,psir_rot[offset+i],ME.J_scheme[i],ME.M_scheme[i],ME.Dimension_J[i]);
                Density_Matrix(psil_v,psir_rot[offset+i],rho_pointer[i],overlap[i]);
            }
            else
            {
                rho_pointer[i] = zeros<cx_mat>(ME.Dimension_M[i],ME.Dimension_M[i]);
                overlap[i] = 1;
            }
        }
        cx_double h1 = Hamiltonian(ME,rho_pointer,overlap);
        cx_double n1 = overlap[0]*overlap[1];

        cx_double h2,n2;
        if(Parity_Decoupling)
        {
            h2 = double(psil_p)*h1;
            n2 = double(psil_p)*n1;
        }
        else
        {
            for(int i=iso-1;i>=0;i--)
            {
                if(MassNumber[i]>0)
                {
                    subview<cx_double> psil_v = psil[i].cols(0,MassNumber[i]-1);
                    Space_Inversion(psir_rot[offset+i],psir_rot[offset+i],ME.M_scheme[i]);
                    Density_Matrix(psil_v,psir_rot[offset+i],rho_pointer[i],overlap[i]);
                }
                else
                {
                    rho_pointer[i] = zeros<cx_mat>(ME.Dimension_M[i],ME.Dimension_M[i]);
                    overlap[i] = 1;
                }
            }
            h2 = Hamiltonian(ME,rho_pointer,overlap);
            n2 = overlap[0]*overlap[1];
        }

        double x = theta_w[be]*phi_w[al]*phi_w[ga]*M_1_PI*M_1_PI*0.125;

        int count_num = 0;
        const int J_size = J.n_elem;
        for(int i=0;i<J_size;i++)
        {
            if(Parity_Decoupling&&(Parity(i)!=psil_p)) continue;

            double j = J(i);
            cx_double c1 = h2*double(Parity(i))+h1;
            cx_double c2 = n2*double(Parity(i))+n1;

            const int degeneracy = int(2*j+1.5);
            const int size_dim = Axial_Symmetry?1:degeneracy;
            const double M_offset = Axial_Symmetry?(j+jmin):0;
            cx_vec v = zeros<cx_vec>(size_dim*size_dim);
            for(int l=size_dim-1;l>=0;l--)
            {
                for(int k=size_dim-1;k>=0;k--)
                {
                    v(l*size_dim+k)=conj(Coef.Wigner_D_Matrix(j,k-j+M_offset,l-j+M_offset,phi_pos[al],theta_pos[be],phi_pos[ga]))*(x*degeneracy);
                }
            }
            H_thread[thread_id].subvec(count_num,count_num+size_dim*size_dim-1) += c1*v;
            N_thread[thread_id].subvec(count_num,count_num+size_dim*size_dim-1) += c2*v;

            count_num+=size_dim*size_dim;
        }
    }

    H.zeros();
    N.zeros();

    for(int i=nthreads-1;i>=0;i--)
    {
        H += H_thread[i];
        N += N_thread[i];
    }
}

template<typename T1, typename T2, typename T3, typename T4>
void HartreeFock::AMP (T1 * psil, T2 * psir, vec & J, ivec & Parity, cx_mat * H, cx_mat * N, T3 * config, T4 & partition)
/** \brief Angular momentum projection
 *
 * \param Bra_WF
 * \param Ket_WF
 * \param J angular momentum
 * \param P parity
 * \return Hamiltonian
 *
 */
{
    static bool flag = true;
    static double * theta_pos;
    static double * theta_w;
    static double * phi_pos;
    static double * phi_w;

    const int iso = Interaction<double>::IsoSpin_Z;
    const int npts = 20;
    const int npts_phi = Axial_Symmetry?1:npts;
    const int totalnum = npts*npts_phi*npts_phi;
    const int nthreads = omp_get_num_procs();
    const int J_size = J.n_elem;
    const double jmin = min(J);
    Wigner_Function Coef;

    cx_mat H_thread[nthreads*J_size];
    cx_mat N_thread[nthreads*J_size];

    if(flag)
    {
        theta_pos = new double [npts];
        theta_w = new double [npts];
        phi_pos = new double [npts_phi];
        phi_w = new double [npts_phi];

        gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc (npts);
        for(int i=npts-1;i>=0;i--)
        {
            gsl_integration_glfixed_point(-1,1,i,&theta_pos[i],&theta_w[i],t);
            theta_pos[i]=acos(theta_pos[i]);
        }
        gsl_integration_glfixed_table_free(t);

        for(int i=npts_phi-1;i>=0;i--)
        {
            phi_w[i]=M_PI*2.0/double(npts_phi);
            phi_pos[i]=i*phi_w[i];
        }

        flag=false;
    }

    for(int i=J_size-1;i>=0;i--)
    {
        double j = J(i);
        const int degeneracy = int(2*j+1.5);
        const int size_dim = Axial_Symmetry?1:degeneracy;
        const int temp = size_dim*partition.n_rows;

        for(int k=nthreads-1;k>=0;k--)
        {
            H_thread[k*J_size+i] = zeros<cx_mat>(temp,temp);
            N_thread[k*J_size+i] = zeros<cx_mat>(temp,temp);
        }
    }

    omp_set_num_threads(nthreads);

    ivec psil_p[iso],psir_p[iso];
    if(Parity_Decoupling)
    {
        for(int i=iso-1;i>=0;i--)
        {
            psil_p[i] = zeros<ivec>(psil[i].n_cols);
            #pragma omp parallel for
            for(int j=psil[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> v = psil[i].col(j);
                psil_p[i](j) = int(Operator_P(v,conj(v),ME.M_scheme[i]).real()+1.5)-1;
            }
            psir_p[i] = zeros<ivec>(psir[i].n_cols);
            #pragma omp parallel for
            for(int j=psir[i].n_cols-1;j>=0;j--)
            {
                subview_col<cx_double> v = psir[i].col(j);
                psir_p[i](j) = int(Operator_P(v,conj(v),ME.M_scheme[i]).real()+1.5)-1;
            }
        }
    }

    #pragma omp parallel for
    for(int loop=totalnum-1;loop>=0;loop--)
    {
        int be = loop/npts_phi/npts_phi;
        int al = (loop/npts_phi)%npts_phi;
        int ga = loop%npts_phi;

        const int thread_id = omp_get_thread_num();
        const int offset = thread_id*J_size;

        cx_mat psir_rot[iso],psir_parity[iso];
        cx_mat over_matrix[iso],over_matrix_parity[iso];
        cx_mat rho_pointer[iso];

        cx_double overlap[iso];

        cx_mat D_matrix[J_size];

        const double x = theta_w[be]*phi_w[al]*phi_w[ga]*M_1_PI*M_1_PI*0.125;
        for(int i=J_size-1;i>=0;i--)
        {
            double j = J(i);
            const int degeneracy = int(2*j+1.5);
            const int size_dim = Axial_Symmetry?1:degeneracy;
            const double M_offset = Axial_Symmetry?(j+jmin):0;
            D_matrix[i] = zeros<cx_mat>(size_dim,size_dim);
            for(int l=size_dim-1;l>=0;l--)
            {
                for(int k=size_dim-1;k>=0;k--)
                {
                    D_matrix[i](k,l)=conj(Coef.Wigner_D_Matrix(j,k-j+M_offset,l-j+M_offset,phi_pos[al],theta_pos[be],phi_pos[ga]))*(x*degeneracy);
                }
            }
        }

        for(int i=iso-1;i>=0;i--)
        {
            if(MassNumber[i]>0)
            {
                psir_rot[i] = zeros<cx_mat>(psir[i].n_rows,psir[i].n_cols);

                Rotation(phi_pos[al],theta_pos[be],phi_pos[ga],psir[i],psir_rot[i],ME.J_scheme[i],ME.M_scheme[i],ME.Dimension_J[i]);
                over_matrix[i] = Overlap_WaveFuction(psil[i],psir_rot[i]);

                if(!Parity_Decoupling)
                {
                    psir_parity[i] = zeros<cx_mat>(psir[i].n_rows,psir[i].n_cols);

                    Space_Inversion(psir_rot[i],psir_parity[i],ME.M_scheme[i]);
                    over_matrix_parity[i] = Overlap_WaveFuction(psil[i],psir_parity[i]);
                }
            }
        }
///!!!!!!!!!!!!!!
        for(int k=partition.n_rows-1;k>=0;k--)
        {
            for(int l=k;l>=0;l--)
            {
                int intrinsic_P = 1;
                bool P_check = true;

                for(int i=iso-1;i>=0;i--)
                {
                    if(MassNumber[i]>0)
                    {
                        uvec index_l = config[i].col(partition(k,i));
                        uvec index_r = config[i].col(partition(l,i));

                        cx_mat psil_v = psil[i].cols(index_l);
                        cx_mat psir_v = psir_rot[i].cols(index_r);
                        cx_mat over_v = over_matrix[i].submat(index_l,index_r);

                        Density_Matrix(psil_v,psir_v,over_v,rho_pointer[i],overlap[i]);

                        if(Parity_Decoupling&&(prod(psil_p[i](index_l))!=(intrinsic_P*=prod(psir_p[i](index_r))))) P_check = false;
                    }
                    else
                    {
                        rho_pointer[i] = zeros<cx_mat>(ME.Dimension_M[i],ME.Dimension_M[i]);
                        overlap[i] = 1;
                    }
                }

                if(!P_check) continue;

                cx_double h1 = Hamiltonian(ME,rho_pointer,overlap);
                cx_double n1 = overlap[0]*overlap[1];

                cx_double h2,n2;
                if(Parity_Decoupling)
                {
                    h2 = double(intrinsic_P)*h1;
                    n2 = double(intrinsic_P)*n1;
                }
                else
                {
                    for(int i=iso-1;i>=0;i--)
                    {
                        if(MassNumber[i]>0)
                        {
                            uvec index_l = config[i].col(partition(k,i));
                            uvec index_r = config[i].col(partition(l,i));

                            cx_mat psil_v = psil[i].cols(index_l);
                            cx_mat psir_v = psir_parity[i].cols(index_r);
                            cx_mat over_v = over_matrix_parity[i].submat(index_l,index_r);

                            Density_Matrix(psil_v,psir_v,over_v,rho_pointer[i],overlap[i]);
                        }
                        else
                        {
                            rho_pointer[i] = zeros<cx_mat>(ME.Dimension_M[i],ME.Dimension_M[i]);
                            overlap[i] = 1;
                        }
                    }
                    h2 = Hamiltonian(ME,rho_pointer,overlap);
                    n2 = overlap[0]*overlap[1];
                }

                for(int i=J_size-1;i>=0;i--)
                {
                    double j = J(i);
                    cx_double c1 = h2*double(Parity(i))+h1;
                    cx_double c2 = n2*double(Parity(i))+n1;

                    const int degeneracy = int(2*j+1.5);
                    const int size_dim = Axial_Symmetry?1:degeneracy;
                    const int index_k = k*size_dim;
                    const int index_l = l*size_dim;

                    H_thread[offset+i].submat(index_k,index_l,size(size_dim,size_dim)) += c1*D_matrix[i];
                    N_thread[offset+i].submat(index_k,index_l,size(size_dim,size_dim)) += c2*D_matrix[i];
                }
            }
        }
    }

    for(int i=J_size-1;i>=0;i--)
    {
        H[i].zeros();
        N[i].zeros();

        for(int k=nthreads-1;k>=0;k--)
        {
            H[i] += H_thread[k*J_size+i];
            N[i] += N_thread[k*J_size+i];
        }
    }
}


void HartreeFock::HW_Equation_Hermitian (cx_mat & A, cx_mat & B, vec & val, cx_mat & wf)
{
    const double EPS = 1e-2;
    const int size_dim = B.n_rows;

    cx_mat psi;
    vec E;
    eig_sym(E,psi,B);

    int count_num =0;
    for(int i=size_dim-1;i>=0;i--)
    {
        if(fabs(E(i))<EPS) break;

        psi.col(i) *= 1.0/sqrt(E(i));
        count_num++;
    }

    if(count_num>0)
    {
        cx_mat A_new = psi.cols(size_dim-count_num,size_dim-1).t()*A*psi.cols(size_dim-count_num,size_dim-1);

        cx_mat psi_new;
        eig_sym(val,psi_new,A_new);
        wf = psi.cols(size_dim-count_num,size_dim-1)*psi_new;
    }
    else
    {
        val=zeros<vec>(0);
    }
}

void HartreeFock::DBP ()
{
    const int nn =1;
    vec J = zeros<vec>(nn);
    ivec P = zeros<ivec>(nn);

    J(0)=6.0;
    P(0)=1;

    int size_dim = int(J(0)*2+1.5);
    cx_mat h = zeros<cx_mat>(size_dim,size_dim);
    cx_mat n = zeros<cx_mat>(size_dim,size_dim);
    cx_vec h_v(h.memptr(),h.n_elem,false);
    cx_vec n_v(n.memptr(),n.n_elem,false);
    cx_mat wf;
    vec E;

    AMP(bra,ket,J,P,h_v,n_v);

    HW_Equation_Hermitian(h,n,E,wf);

    cout<<"Energy levels:"<<endl;
    for(int i=E.n_elem-1;i>=0;i--)
    {
        cout<<i<<" "<<E(i)<<endl;
    }

}

template<typename T1, typename T>
T HartreeFock::Construct_ME (Interaction<T> & Force, T1 * psi_l, T1 * psi_r)
{
    const size_t Nbody = Interaction<T>::N_Body;
    const int iso = Interaction<T>::IsoSpin_Z;

    T res = T(0);

    size_t count_num =0;
    for(int loop=iso-1;loop>=0;loop--)
    {
        count_num+=(psi_l[loop]^psi_r[loop]).count();
    }
    if((count_num>>=1)>Nbody) return res;

    //pp, nn interaction
    for(int loop=iso-1;loop>=0;loop--)
    {
        const int pn = iso-loop-1;
        if((psi_l[pn]^psi_r[pn]).any()) continue;

        for(int n = Force.OBME[loop].n_elem-1;n>=0;n--)
        {
            int pos = n/Force.OBME[loop].n_cols;
            if(psi_r[loop][pos]&&(psi_l[loop]^(Force.Op_OB[loop][n]^psi_r[loop])).none())
            {
                res += (Force.Phase_OB[loop][n]&psi_r[loop]).count()&1?(-Force.OBME[loop](n)):Force.OBME[loop](n);
            }
        }


        umat * pointer = &Force.index[loop];
        vector<bitset<Interaction<T>::Max_Bit> > * p_bit = &Force.Op_TB[loop];
        vector<bitset<Interaction<T>::Max_Bit> > * phase_bit = &Force.Phase_TB[loop];
        for(int n = pointer->n_cols-1;n>=0;n--)
        {
            subview<uword> col = pointer->col(n);
            if(psi_r[loop][col(2)]&&psi_r[loop][col(3)])
            {
                if((psi_l[loop]^((*p_bit)[n]^psi_r[loop])).none())
                {
                    res+= ((*phase_bit)[n]&psi_r[loop]).count()&1?(-Force.TBME[loop](n)):(Force.TBME[loop](n));
                }
            }
        }
    }

    //pn interaction
    umat * pointer = &Force.index[iso];
    for(int n = pointer->n_cols-1;n>=0;n--)
    {
        subview<uword> col = pointer->col(n);
        bool flag = true;

        int phase = 1;
        for(int loop=iso-1;loop>=0;loop--)
        {
            if(!(psi_r[loop][col(loop+iso)]&&(psi_l[loop]^(Force.Op_TB_pn[loop][n]^psi_r[loop])).none()))
            {
                flag = false;
                break;
            }
            phase *= (Force.Phase_TB_pn[loop][n]&psi_r[loop]).count()&1?-1:1;
        }

        if(flag) res+=Force.TBME[iso](n)*double(phase);
    }

    return res;
}

template<typename T1, typename T>
T HartreeFock::Construct_ME2 (Interaction<T> & Force, T1 * psi_l, T1 * psi_r)
{
    const size_t Nbody = Interaction<T>::N_Body;
    const int iso = Interaction<T>::IsoSpin_Z;

    T res = T(0);

    unsigned int count_num = 0;
    uvec count_num_iso = zeros<uvec>(iso);
    for(int loop=iso-1;loop>=0;loop--)
    {
        count_num_iso(loop) = (psi_l[loop]^psi_r[loop]).count();
        count_num += count_num_iso(loop);
    }
    if((count_num>>=1)>Nbody) return res;


    switch(count_num)
    {
        case 0:
        {
            //pp, nn interaction
            for(int loop=iso-1;loop>=0;loop--)
            {
                const int pn = iso-loop-1;
                if((psi_l[pn]^psi_r[pn]).any()) continue;

                for(int n = Force.OBME[loop].n_elem-1;n>=0;n--)
                {
                    int pos = n/Force.OBME[loop].n_cols;
                    if(psi_r[loop][pos]&&(psi_l[loop]^(Force.Op_OB[loop][n]^psi_r[loop])).none())
                    {
                        res += (Force.Phase_OB[loop][n]&psi_r[loop]).count()&1?(-Force.OBME[loop](n)):Force.OBME[loop](n);
                    }
                }


                umat * pointer = &Force.index[loop];
                vector<bitset<Interaction<T>::Max_Bit> > * p_bit = &Force.Op_TB[loop];
                vector<bitset<Interaction<T>::Max_Bit> > * phase_bit = &Force.Phase_TB[loop];
                for(int n = pointer->n_cols-1;n>=0;n--)
                {
                    subview<uword> col = pointer->col(n);
                    if(psi_r[loop][col(2)]&&psi_r[loop][col(3)])
                    {
                        if((psi_l[loop]^((*p_bit)[n]^psi_r[loop])).none())
                        {
                            res+= ((*phase_bit)[n]&psi_r[loop]).count()&1?(-Force.TBME[loop](n)):(Force.TBME[loop](n));
                        }
                    }
                }
            }

            //pn interaction
            umat * pointer = &Force.index[iso];
            for(int n = pointer->n_cols-1;n>=0;n--)
            {
                subview<uword> col = pointer->col(n);
                bool flag = true;

                int phase = 1;
                for(int loop=iso-1;loop>=0;loop--)
                {
                    if(!(psi_r[loop][col(loop+iso)]&&(psi_l[loop]^(Force.Op_TB_pn[loop][n]^psi_r[loop])).none()))
                    {
                        flag = false;
                        break;
                    }
                    phase *= (Force.Phase_TB_pn[loop][n]&psi_r[loop]).count()&1?-1:1;
                }

                if(flag) res+=Force.TBME[iso](n)*double(phase);
            }

            break;
        }
        case 1:
        {
            //pp, nn interaction
            for(int loop=iso-1;loop>=0;loop--)
            {
                const int pn = iso-loop-1;
                if((psi_l[pn]^psi_r[pn]).any()) continue;

                for(int n = Force.OBME[loop].n_elem-1;n>=0;n--)
                {
                    int pos = n/Force.OBME[loop].n_cols;
                    if(psi_r[loop][pos]&&(psi_l[loop]^(Force.Op_OB[loop][n]^psi_r[loop])).none())
                    {
                        res += (Force.Phase_OB[loop][n]&psi_r[loop]).count()&1?(-Force.OBME[loop](n)):Force.OBME[loop](n);
                    }
                }


                umat * pointer = &Force.index[loop];
                vector<bitset<Interaction<T>::Max_Bit> > * p_bit = &Force.Op_TB[loop];
                vector<bitset<Interaction<T>::Max_Bit> > * phase_bit = &Force.Phase_TB[loop];
                for(int n = pointer->n_cols-1;n>=0;n--)
                {
                    subview<uword> col = pointer->col(n);
                    if(psi_r[loop][col(2)]&&psi_r[loop][col(3)])
                    {
                        if((psi_l[loop]^((*p_bit)[n]^psi_r[loop])).none())
                        {
                            res+= ((*phase_bit)[n]&psi_r[loop]).count()&1?(-Force.TBME[loop](n)):(Force.TBME[loop](n));
                        }
                    }
                }
            }

            //pn interaction
            umat * pointer = &Force.index[iso];
            for(int n = pointer->n_cols-1;n>=0;n--)
            {
                subview<uword> col = pointer->col(n);
                bool flag = true;

                int phase = 1;
                for(int loop=iso-1;loop>=0;loop--)
                {
                    if(!(psi_r[loop][col(loop+iso)]&&(psi_l[loop]^(Force.Op_TB_pn[loop][n]^psi_r[loop])).none()))
                    {
                        flag = false;
                        break;
                    }
                    phase *= (Force.Phase_TB_pn[loop][n]&psi_r[loop]).count()&1?-1:1;
                }

                if(flag) res+=Force.TBME[iso](n)*double(phase);
            }

            break;
        }
        case 2:
        {
            //pp, nn interaction
            for(int loop=iso-1;loop>=0;loop--)
            {
                const int pn = iso-loop-1;
                if((psi_l[pn]^psi_r[pn]).any()) continue;

                for(int n = Force.OBME[loop].n_elem-1;n>=0;n--)
                {
                    int pos = n/Force.OBME[loop].n_cols;
                    if(psi_r[loop][pos]&&(psi_l[loop]^(Force.Op_OB[loop][n]^psi_r[loop])).none())
                    {
                        res += (Force.Phase_OB[loop][n]&psi_r[loop]).count()&1?(-Force.OBME[loop](n)):Force.OBME[loop](n);
                    }
                }


                umat * pointer = &Force.index[loop];
                vector<bitset<Interaction<T>::Max_Bit> > * p_bit = &Force.Op_TB[loop];
                vector<bitset<Interaction<T>::Max_Bit> > * phase_bit = &Force.Phase_TB[loop];
                for(int n = pointer->n_cols-1;n>=0;n--)
                {
                    subview<uword> col = pointer->col(n);
                    if(psi_r[loop][col(2)]&&psi_r[loop][col(3)])
                    {
                        if((psi_l[loop]^((*p_bit)[n]^psi_r[loop])).none())
                        {
                            res+= ((*phase_bit)[n]&psi_r[loop]).count()&1?(-Force.TBME[loop](n)):(Force.TBME[loop](n));
                        }
                    }
                }
            }

            //pn interaction
            umat * pointer = &Force.index[iso];
            for(int n = pointer->n_cols-1;n>=0;n--)
            {
                subview<uword> col = pointer->col(n);
                bool flag = true;

                int phase = 1;
                for(int loop=iso-1;loop>=0;loop--)
                {
                    if(!(psi_r[loop][col(loop+iso)]&&(psi_l[loop]^(Force.Op_TB_pn[loop][n]^psi_r[loop])).none()))
                    {
                        flag = false;
                        break;
                    }
                    phase *= (Force.Phase_TB_pn[loop][n]&psi_r[loop]).count()&1?-1:1;
                }

                if(flag) res+=Force.TBME[iso](n)*double(phase);
            }

            break;
        }
    }
    //pp, nn interaction
    for(int loop=iso-1;loop>=0;loop--)
    {
        const int pn = iso-loop-1;
        if((psi_l[pn]^psi_r[pn]).any()) continue;

        for(int n = Force.OBME[loop].n_elem-1;n>=0;n--)
        {
            int pos = n/Force.OBME[loop].n_cols;
            if(psi_r[loop][pos]&&(psi_l[loop]^(Force.Op_OB[loop][n]^psi_r[loop])).none())
            {
                res += (Force.Phase_OB[loop][n]&psi_r[loop]).count()&1?(-Force.OBME[loop](n)):Force.OBME[loop](n);
            }
        }


        umat * pointer = &Force.index[loop];
        vector<bitset<Interaction<T>::Max_Bit> > * p_bit = &Force.Op_TB[loop];
        vector<bitset<Interaction<T>::Max_Bit> > * phase_bit = &Force.Phase_TB[loop];
        for(int n = pointer->n_cols-1;n>=0;n--)
        {
            subview<uword> col = pointer->col(n);
            if(psi_r[loop][col(2)]&&psi_r[loop][col(3)])
            {
                if((psi_l[loop]^((*p_bit)[n]^psi_r[loop])).none())
                {
                    res+= ((*phase_bit)[n]&psi_r[loop]).count()&1?(-Force.TBME[loop](n)):(Force.TBME[loop](n));
                }
            }
        }
    }

    //pn interaction
    umat * pointer = &Force.index[iso];
    for(int n = pointer->n_cols-1;n>=0;n--)
    {
        subview<uword> col = pointer->col(n);
        bool flag = true;

        int phase = 1;
        for(int loop=iso-1;loop>=0;loop--)
        {
            if(!(psi_r[loop][col(loop+iso)]&&(psi_l[loop]^(Force.Op_TB_pn[loop][n]^psi_r[loop])).none()))
            {
                flag = false;
                break;
            }
            phase *= (Force.Phase_TB_pn[loop][n]&psi_r[loop]).count()&1?-1:1;
        }

        if(flag) res+=Force.TBME[iso](n)*double(phase);
    }

    return res;
}

template <typename T>
void HartreeFock::Deformation (Interaction<T> & Force, cx_mat* rho, double &dbeta, double &dgamma)
{
    const int dim = 3;
    const int dim_q0 = 6;
    const double cons1 = sqrt(6./5.);
    const double cons2 = 1.0/sqrt(5.);

    const int iso = Interaction<T>::IsoSpin_Z;
    const unsigned int n_states = Force.Dimension_M[0];

    double q1, q2;
    vec q_diag;
    cx_double trace_v,r2trace;

    cx_mat q(dim,dim),qp(dim,dim),qn(dim,dim);
    mat r2 = zeros<mat>(n_states,n_states);
    cube q0 = zeros<cube>(n_states,n_states,dim_q0);

    #pragma omp parallel for
    for (int i=n_states-1;i>=0;i--)
    {
        int l1=Force.M_scheme[0][i].l;
        double j1=Force.M_scheme[0][i].j;
        int ij1=int(j1*2.0+0.5);
        int n1=Force.M_scheme[0][i].n;
        double m1=Force.M_scheme[0][i].m;
        int im1=int(m1*2.0+0.5+ij1)-ij1;

        int sign=((ij1-im1)>>1)&1?-1:1;

        for (int j=n_states-1;j>=0;j--)
        {
            int l2=Force.M_scheme[0][j].l;
            double j2=Force.M_scheme[0][j].j;
            int ij2=int(j2*2.0+0.5);
            int n2=Force.M_scheme[0][j].n;
            double m2=Force.M_scheme[0][j].m;
            int im2=int(m2*2.0+0.5+ij2)-ij2;

            double delta=0.0;
            if((im1==im2) && (ij1==ij2) && (l1==l2)) delta=1.0;

            double fact = sign*YredME(2,l1,j1,l2,j2)*0.5;

            double f1=fact*gsl_sf_coupling_3j(ij1,4,ij2,-im1,-4,im2);
            double f2=fact*gsl_sf_coupling_3j(ij1,4,ij2,-im1,-2,im2);
            double f3=fact*gsl_sf_coupling_3j(ij1,4,ij2,-im1, 0,im2);
            double f4=fact*gsl_sf_coupling_3j(ij1,4,ij2,-im1,2,im2);
            double f5=fact*gsl_sf_coupling_3j(ij1,4,ij2,-im1,4,im2);

            double f6=HORadInt(2,n1,l1,n2,l2);
            r2(i,j)=f6*delta;

            q0(i,j,0)= f6*((f1+f5)*cons1-2.*f3*cons2);        // xx
            q0(i,j,1)=-f6*((f1+f5)*cons1+2.*f3*cons2);        // yy
            q0(i,j,2)= 4.*f6*f3*cons2;                        // zz
            q0(i,j,3)= f6*cons1*(f1-f5); // *I (purely imaginary)xy
            q0(i,j,4)=-f6*cons1*(f4-f2);                      // xz
            q0(i,j,5)= f6*cons1*(f4+f2); // *I (purely imaginary)yz

        }
    }

    imat map2 = zeros<imat>(6,iso);
    for (int m=iso;m>=0;m--)
    {
        map2(m,0)=m;
        map2(m,1)=m;
    }
    map2(3,0)=0;
    map2(3,1)=1;
    map2(4,0)=0;
    map2(4,1)=2;
    map2(5,0)=1;
    map2(5,1)=2;

    mat tmp = zeros<mat>(n_states,n_states);

    cx_mat * rhop = &rho[0];
    cx_mat *rhon = &rho[1];

    for (int m=dim_q0-1;m>=0;m--)
    {
        trace_v = compute_trace(q0.slice(m),*rhop);
        qp(map2(m,0),map2(m,1))=trace_v;
        qp(map2(m,1),map2(m,0))=trace_v;
        trace_v = compute_trace(q0.slice(m),*rhon);
        qn(map2(m,0),map2(m,1))=trace_v;
        qn(map2(m,1),map2(m,0))=trace_v;
    }

    trace_v = compute_trace(r2,*rhop);
    r2trace = compute_trace(r2,*rhon);
    r2trace += trace_v;

    qp.diag(1).zeros();
    qp.diag(-1).zeros();
    qn.diag(1).zeros();
    qn.diag(-1).zeros();

    q = qp+qn;
    q_diag=eig_sym(q);

    intrinsic(q_diag,q1,q2);

    dgamma=atan(q2*sqrt(2.)/q1)*180.*M_1_PI;
    if(dgamma<0.0) dgamma=-dgamma;

    dbeta=norm(q_diag);
    dbeta*=dbeta;
    dbeta=5.*sqrt(dbeta/6.)/6.;
    dbeta=real(dbeta/r2trace);
}

template<typename T>
cx_mat HartreeFock::Q20_operator(Interaction<T> & Force)
{
    const unsigned int n_states = Force.Dimension_M[0];

    cx_mat q20=zeros<cx_mat>(n_states,n_states);

    #pragma omp parallel for
    for (int i=n_states-1;i>=0;i--)
    {
        int l1=Force.M_scheme[0][i].l;
        double j1=Force.M_scheme[0][i].j;
        int ij1=int(j1*2.0+0.5);
        int n1=Force.M_scheme[0][i].n;
        double m1=Force.M_scheme[0][i].m;
        int im1=int(m1*2.0+0.5+ij1)-ij1;

        int sign=((ij1-im1)>>1)&1?-1:1;

        for (int j=n_states-1;j>=0;j--)
        {
            int l2=Force.M_scheme[0][j].l;
            double j2=Force.M_scheme[0][j].j;
            int ij2=int(j2*2.0+0.5);
            int n2=Force.M_scheme[0][j].n;
            double m2=Force.M_scheme[0][j].m;
            int im2=int(m2*2.0+0.5+ij2)-ij2;

            double f3=sign*gsl_sf_coupling_3j(ij1,4,ij2,-im1, 0,im2);
            f3*=YredME(2,l1,j1,l2,j2)*0.5;

            double f6=HORadInt(2,n1,l1,n2,l2);
            q20(i,j)= 4.*f6*f3/sqrt(5.);
        }
    }

    return q20;
}

void HartreeFock :: Set_HF_Basis()
{
	for(int iso=0; iso<=Interaction<double>::IsoSpin_Z-1; iso++)
	{
		hf_basis[iso] = new HF_SPS [ME.Dimension_M[iso]];
		for(int i=0; i<=ME.Dimension_M[iso]-1; i++)
		{
			int par = -1;
			double m = 1e3, spe;
			for(int j=0; j<=ME.Dimension_M[iso]-1; j++)
			{
				if(abs(bra[iso](j,i))>=1e-2)
				{
					int partmp;
					partmp = ME.M_scheme[iso][j].l%2==1?1:0;
					if(m == 1e3) m = ME.M_scheme[iso][j].m;
					else if(fabs(m - ME.M_scheme[iso][j].m)>=1e-3) cout << "M mixed!" << "  " << m << "  " << ME.M_scheme[iso][j].m << endl;
					if(par == -1) par = partmp;
					else if(par != partmp) cout << "Parity mixed!" << endl;
				}
			}
			spe = E_sp[iso](i);
			hf_basis[iso][i].set(i, iso-0.5, par, m, spe);
		}
	}
}

void HartreeFock :: Save_HF_Basis(string filename)
{
	ofstream hfsps(filename.c_str());
	hfsps << "   index    isospin    parity      m      energy" << endl;
	for(int iso=0; iso<=Interaction<double>::IsoSpin_Z-1; iso++)
	  for(int i=0; i<=ME.Dimension_M[iso]-1; i++)
	  {
		  hfsps << hf_basis[iso][i].index << "  " << hf_basis[iso][i].Iz << "  " << hf_basis[iso][i].par << "  " << hf_basis[iso][i].m << "  " << hf_basis[iso][i].spe << endl;
	  }
	hfsps << endl;
	hfsps.close();
}


/***********************************************************************************************************************/
class Nucleus
{
public:
    Nucleus(int Z, int A, string space, string force, double bet);
    ~Nucleus() {}

    ivec Construct_Configuration (vec & E_cutoff, int PH_limit);
    void Spectra (vec & J, ivec & Parity);
    void Spectra_MemOptimize (vec & J, ivec & Parity);
    void Spectra ();
    void Spectra_GCM ();

private:
    HartreeFock Basis;

    umat Configuration[Interaction<double>::IsoSpin_Z];
    vec Configuration_M[Interaction<double>::IsoSpin_Z];
    vec Excitation_E[Interaction<double>::IsoSpin_Z];
};

Nucleus::Nucleus(int Z, int A, string space, string force, double bet):Basis(Z,A,space,force,bet)
{
}

ivec Nucleus::Construct_Configuration (vec & E_cutoff, int PH_limit = 0)
/** \brief Construct different configurations
 *
 * \param WF Hartree Fock wavefunctions
 * \return Different excited configurations
 *
 */
{
    const int iso = Interaction<double>::IsoSpin_Z;
    ivec dim=zeros<ivec>(iso);

    double Fermi;
    for(int i=0;i<iso;i++)
    {
        if(Basis.MassNumber[i]<=0) continue;

        vec * pointer = &Basis.E_sp[i];

        const int size_dim = pointer->n_elem;
        double cutoff = E_cutoff(i);
        int up,low;

        Fermi = (*pointer)(Basis.MassNumber[i]-1);
        for(up=Basis.MassNumber[i];up<size_dim;up++)
        {
            if(fabs((*pointer)(up)-Fermi)>cutoff) break;
        }
        dim(i) = up;
        Fermi = (*pointer)(Basis.MassNumber[i]);
        for(low=Basis.MassNumber[i]-1;low>=0;low--)
        {
            if(fabs((*pointer)(low)-Fermi)>cutoff) break;
        }

        if(low==(Basis.MassNumber[i]-1)) low--;
        const unsigned int particle = Basis.MassNumber[i]-low-1;
        const unsigned int offset = low+1;
        const unsigned int offset_up = Basis.MassNumber[i];

        cout<<"The energy cutoff for "<<i<<" is "<<cutoff<<" MeV"<<endl;
        cout<<"There are "<<up-Basis.MassNumber[i]<<" levels above Fermi surface and "<<Basis.MassNumber[i]-low-1<<" levels below."<<endl;

        int count_num =0;
        if(PH_limit<=0)
        {
            gsl_combination * Config = gsl_combination_calloc(up-low-1,particle);

            Fermi = 0.0;
	    int cuttot = 0;
            for(int j=particle-1;j>=0;j--)
            {
                if(cuttot==1) Fermi += ((*pointer)(offset+j)+(Basis.ME.M_scheme[i][offset+j].n+0.5)*Basis.ME.hbar_omega/2.)/2.;
		else Fermi += (*pointer)(offset+j);
            }

            ivec temp = zeros<ivec>(int(gsl_sf_choose(Config->n,particle)+0.5));
            vec temp_E = zeros<vec>(temp.n_elem);

            int k=0;
            do
            {
                double sum = 0.0;
                for(int j=particle-1;j>=0;j--)
                {
		    if(cuttot==1) sum += ((*pointer)(offset+Config->data[j])+(Basis.ME.M_scheme[i][offset+Config->data[j]].n+0.5)*Basis.ME.hbar_omega/2.)/2.;
                    else sum +=(*pointer)(offset+Config->data[j]);
		}
                if(fabs(sum-Fermi)<=cutoff)
                {
                    temp(count_num) = k;
                    temp_E(count_num) = sum-Fermi;
                    count_num++;
                }
                k++;
            }
            while(gsl_combination_next(Config) == GSL_SUCCESS);

            Configuration[i] = zeros<umat>(count_num,particle);
            if(count_num>0) Excitation_E[i] = temp_E.subvec(0,count_num-1);

            cout<<"Number of configurations is "<<count_num<<endl;

            int count_num2 = 0;
            k = 0;
            gsl_combination_init_first(Config);
            do
            {
                if(count_num2>=count_num) break;
                if(k==temp(count_num2))
                {
                    for(int j=particle-1;j>=0;j--)
                    {
                        Configuration[i](count_num2,j)=offset+Config->data[j];
                    }
                    count_num2++;
                }
                k++;
            }
            while(gsl_combination_next(Config) == GSL_SUCCESS);

            gsl_combination_free(Config);
        }
        else
        {
            int up_dim = up-Basis.MassNumber[i];
            int low_dim = Basis.MassNumber[i]-low-1;
            int limit = min(min(up_dim,low_dim),PH_limit);

            int dim =0;
            for(int loop=limit;loop>=0;loop--)
            {
                dim += int(gsl_sf_choose(low_dim,loop)*gsl_sf_choose(up_dim,loop)+0.5);
            }

            ivec temp = zeros<ivec>(dim);
            vec temp_E = zeros<vec>(temp.n_elem);

            int k=0;

            for(int loop=limit;loop>=0;loop--)
            {
                gsl_combination * Config_up = gsl_combination_calloc(up_dim,loop);
                gsl_combination * Config_low = gsl_combination_calloc(low_dim,loop);

                do
                {
                    double sum_low = 0.0;
                    for(int j=loop-1;j>=0;j--)
                    {
                        sum_low -= (*pointer)(offset+Config_low->data[j]);
                    }

                    gsl_combination_init_first(Config_up);
                    do
                    {
                        double sum_up = 0.0;
                        for(int j=loop-1;j>=0;j--)
                        {
                            sum_up += (*pointer)(offset_up+Config_up->data[j]);
                        }
                        if(fabs(sum_up+sum_low)<=cutoff)
                        {
                            temp(count_num) = k;
                            temp_E(count_num) = sum_up+sum_low;
                            count_num++;
                        }
                        k++;
                    }
                    while(gsl_combination_next(Config_up) == GSL_SUCCESS);
                }
                while(gsl_combination_next(Config_low) == GSL_SUCCESS);

                gsl_combination_free(Config_low);
                gsl_combination_free(Config_up);
            }

            Configuration[i] = zeros<umat>(count_num,particle);
            if(count_num>0) Excitation_E[i] = temp_E.subvec(0,count_num-1);

            cout<<"Number of configurations is "<<count_num<<endl;

            int count_num2 = 0;
            k = 0;

            urowvec aux = zeros<urowvec>(particle);
            for(int loop=particle-1;loop>=0;loop--) aux(loop) = offset+loop;

            for(int loop=limit;loop>=0;loop--)
            {
                gsl_combination * Config_up = gsl_combination_calloc(up_dim,loop);
                gsl_combination * Config_low = gsl_combination_calloc(low_dim,loop);

                do
                {
                    if(count_num2>=count_num) break;
                    gsl_combination_init_first(Config_up);
                    do
                    {
                        if(k==temp(count_num2))
                        {
                            Configuration[i].row(count_num2) = aux;
                            for(int j=loop-1;j>=0;j--)
                            {
                                Configuration[i](count_num2,Config_low->data[j])=offset_up+Config_up->data[j];
                            }
                            count_num2++;
                        }
                        k++;
                    }
                    while(gsl_combination_next(Config_up) == GSL_SUCCESS);
                }
                while(gsl_combination_next(Config_low) == GSL_SUCCESS);

                gsl_combination_free(Config_low);
                gsl_combination_free(Config_up);
            }

            sort(Configuration[i],"ascend",1);
        }

        if(HartreeFock::Axial_Symmetry)
        {
            Configuration_M[i] = zeros<vec>(count_num);

            double offset_M = 0.0;
            for(int j=offset-1;j>=0;j--)
            {
                offset_M += Basis.M_record[i](j);
            }

            for(int k=count_num-1;k>=0;k--)
            {
                double sum = 0.0;
                for(int j=particle-1;j>=0;j--)
                {
                    sum += Basis.M_record[i](Configuration[i](k,j));
                }

                Configuration_M[i](k) = sum+offset_M;
            }
        }
    }

    return dim;
}

///!!!!!
double cutoff[Interaction<double>::IsoSpin_Z] = {20,20};
void Nucleus::Spectra (vec & J, ivec & Parity)
/** \brief Spectra with angular momentum projection
 *
 * \param J angular momentum
 * \param P parity
 * \return Excited energies
 *
 */
{
    const unsigned int Particle_Hole_Truncation = 0;

    const int iso = Interaction<double>::IsoSpin_Z;
    const unsigned int Levels_show = 3;
    const int J_size = J.n_elem;
    ///!!!!!!double cutoff[iso] = {30,0};

    vec E_cutoff(cutoff,iso);

    Basis.Iteration(Basis.ME);
    ///!!!Basis.Iteration_BHF(Basis.ME);

    ivec dim_max = Construct_Configuration(E_cutoff,Particle_Hole_Truncation);

    cx_mat H[J_size];
    cx_mat N[J_size];
    cx_mat wf[J_size];
    vec E [J_size];
    umat Config[iso];

    cx_mat bra_sub[iso],ket_sub[iso];
    for(int i=iso-1;i>=0;i--)
    {
        if(Configuration[i].is_empty()) continue;

        Config[i] = zeros<umat>(Basis.MassNumber[i],Configuration[i].n_rows);
        Config[i].rows(Basis.MassNumber[i]-Configuration[i].n_cols,Basis.MassNumber[i]-1) = Configuration[i].t();
        for(int n=Basis.MassNumber[i]-Configuration[i].n_cols-1;n>=0;n--)
        {
            for(int m=Configuration[i].n_rows-1;m>=0;m--) Config[i](n,m) = n;
        }

        if((Basis.bra[i].n_cols>0)&&(Basis.ket[i].n_cols>0)&&(dim_max(i)>0))
        {
            bra_sub[i] = Basis.bra[i].cols(0,dim_max(i)-1);
            ket_sub[i] = Basis.ket[i].cols(0,dim_max(i)-1);
        }
    }

    const int size1 = max(int(Configuration[0].n_rows),1);
    const int size2 = max(int(Configuration[1].n_rows),1);

    imat Combination_duplicate = zeros<imat>(size1*size2,iso);
    int dim = 0;
    double jmin = min(J);
    for(int i=size1-1;i>=0;i--)
    {
        for(int j=size2-1;j>=0;j--)
        {
            if(HartreeFock::Axial_Symmetry)
            {
                double temp = Configuration[0].is_empty()?0:Configuration_M[0](i);
                temp += Configuration[1].is_empty()?0:Configuration_M[1](j);
                if(int(fabs(temp-jmin)+0.5)==0)
                {
                    if(Particle_Hole_Truncation)
                    {
                        uvec q1,q2;
                        if(!Configuration[0].is_empty()) q1=find(Configuration[0].row(i)>=Basis.MassNumber[0]);
                        if(!Configuration[1].is_empty()) q2=find(Configuration[1].row(j)>=Basis.MassNumber[1]);

                        ///!!!if((q1.n_elem+q2.n_elem)==Particle_Hole_Truncation||(q1.n_elem+q2.n_elem)==0)
                        if((q1.n_elem+q2.n_elem)<=Particle_Hole_Truncation)
                        {
                            Combination_duplicate(dim,0) = i;
                            Combination_duplicate(dim++,1) = j;
                        }
                    }
                    else
                    {
                        Combination_duplicate(dim,0) = i;
                        Combination_duplicate(dim++,1) = j;
                    }
                }
            }
            else
            {
                if(Particle_Hole_Truncation)
                {
                    uvec q1,q2;
                    if(!Configuration[0].is_empty()) q1=find(Configuration[0].row(i)>=Basis.MassNumber[0]);
                    if(!Configuration[1].is_empty()) q2=find(Configuration[1].row(j)>=Basis.MassNumber[1]);

                    ///!!!if((q1.n_elem+q2.n_elem)==Particle_Hole_Truncation||(q1.n_elem+q2.n_elem)==0)
                    if((q1.n_elem+q2.n_elem)<=Particle_Hole_Truncation)
                    {
                        Combination_duplicate(dim,0) = i;
                        Combination_duplicate(dim++,1) = j;
                    }
                }
                else
                {
                    Combination_duplicate(dim,0) = i;
                    Combination_duplicate(dim++,1) = j;
                }
            }
        }
    }
    cout<<"Total dimension is "<<dim<<endl;
    imat Combination;
    if(dim>0) Combination = Combination_duplicate.submat(0,0,dim-1,iso-1);


    for(int loop=0;loop<J_size;loop++)
    {
        const int i = HartreeFock::Axial_Symmetry?1:int(J(loop)*2+1.5);
        const int dim_temp = i*dim;

        H[loop] = zeros<cx_mat>(dim_temp,dim_temp);
        N[loop] = zeros<cx_mat>(dim_temp,dim_temp);
    }

    cout<<"Construct matrix elements among different configurations."<<endl;
    Basis.AMP(bra_sub,ket_sub,J,Parity,H,N,Config,Combination);

    ofstream ofile("Spectra.txt");
    ofile<<"The spectral of nucleus Z = "<<Basis.MassNumber[0]<<", A = "<<Basis.MassNumber[iso]<<endl;
    cout<<"The spectral of nucleus Z = "<<Basis.MassNumber[0]<<", A = "<<Basis.MassNumber[iso]<<endl;
    for(int loop=J.n_elem-1;loop>=0;loop--)
    {
        for(int i=H[loop].n_rows-1;i>=0;i--)
        {
            for(int j=i-1;j>=0;j--)
            {
                H[loop](j,i)=conj(H[loop](i,j));
                N[loop](j,i)=conj(N[loop](i,j));
            }
        }

        Basis.HW_Equation_Hermitian(H[loop],N[loop],E[loop],wf[loop]);

        ofile<<"J: "<<J(loop)<<", P: "<<Parity(loop)<<endl;
        for(int i = min(Levels_show,E[loop].n_elem)-1;i>=0;i--)
        {
            ofile<<i<<" "<<E[loop](i)<<endl;
            cout<<i<<", J: "<<J(loop)<<", P: "<<Parity(loop)<<", Energy: "<<E[loop](i)<<" MeV"<<endl;
        }
    }

    ofile.close();
}

void Nucleus::Spectra_MemOptimize (vec & J, ivec & Parity)
/** \brief Spectra with angular momentum projection
 *
 * \param J angular momentum
 * \param P parity
 * \return Excited energies
 *
 */
{
    const int iso = Interaction<double>::IsoSpin_Z;
    const unsigned int Levels_show = 3;
    const int J_size = J.n_elem;
    ///!!!!!!double cutoff[iso] = {30,0};

    vec E_cutoff(cutoff,iso);

    Basis.Iteration(Basis.ME);

    Construct_Configuration(E_cutoff);

    const int size1 = max(int(Configuration[0].n_rows),1);
    const int size2 = max(int(Configuration[1].n_rows),1);
    int offset[iso];

    ivec blocksize = zeros<ivec>(J_size);

    cx_mat H[J_size];
    cx_mat N[J_size];
    cx_mat wf[J_size];
    vec E[J_size];

    cx_mat bra[iso];
    cx_mat ket[iso];

    for(int i=iso-1;i>=0;i--)
    {
        offset[i] = Basis.MassNumber[i]-Configuration[i].n_cols;

        if(Basis.MassNumber[i]<=0) continue;

        bra[i] = zeros<cx_mat>(Basis.bra[i].n_rows,Basis.MassNumber[i]);
        ket[i] = zeros<cx_mat>(Basis.ket[i].n_rows,Basis.MassNumber[i]);

        if(offset[i]>0)
        {
            bra[i].cols(0,offset[i]-1) = Basis.bra[i].cols(0,offset[i]-1);
            ket[i].cols(0,offset[i]-1) = Basis.ket[i].cols(0,offset[i]-1);
        }
    }


    imat Combination = zeros<imat>(size1*size2,iso);
    int dim = 0;
    double jmin = min(J);
    for(int i=size1-1;i>=0;i--)
    {
        for(int j=size2-1;j>=0;j--)
        {
            if(HartreeFock::Axial_Symmetry)
            {
                double temp = Configuration[0].is_empty()?0:Configuration_M[0](i);
                temp += Configuration[1].is_empty()?0:Configuration_M[1](j);
                if(int(fabs(temp-jmin)+0.5)==0)
                {
                    Combination(dim,0) = i;
                    Combination(dim++,1) = j;
                }
            }
            else
            {
                Combination(dim,0) = i;
                Combination(dim++,1) = j;
            }
        }
    }
    cout<<"Total dimension is "<<dim<<endl;

    int count_num = 0;
    for(int loop=0;loop<J_size;loop++)
    {
        const int i = HartreeFock::Axial_Symmetry?1:int(J(loop)*2+1.5);
        const int dim_temp = i*dim;

        blocksize(loop)=i;
        count_num += i*i;

        H[loop] = zeros<cx_mat>(dim_temp,dim_temp);
        N[loop] = zeros<cx_mat>(dim_temp,dim_temp);
    }

    cx_vec h_v = zeros<cx_vec>(count_num);
    cx_vec n_v = zeros<cx_vec>(count_num);

    cout<<"Construct matrix elements among different configurations."<<endl;


    for(int i=dim-1;i>=0;i--)
    {
        const int index_bra_p = Combination(i,0);
        for(int ii=Configuration[0].n_cols-1;ii>=0;ii--)
        {
            bra[0].col(offset[0]+ii) = Basis.bra[0].col(Configuration[0](index_bra_p,ii));
        }

        const int index_bra_n = Combination(i,1);
        for(int ii=Configuration[1].n_cols-1;ii>=0;ii--)
        {
            bra[1].col(offset[1]+ii) = Basis.bra[1].col(Configuration[1](index_bra_n,ii));
        }

        for(int j=i;j>=0;j--)
        {
            const int index_ket_p = Combination(j,0);
            for(int jj=Configuration[0].n_cols-1;jj>=0;jj--)
            {
                ket[0].col(offset[0]+jj) = Basis.ket[0].col(Configuration[0](index_ket_p,jj));
            }

            const int index_ket_n = Combination(j,1);
            for(int jj=Configuration[1].n_cols-1;jj>=0;jj--)
            {
                ket[1].col(offset[1]+jj) = Basis.ket[1].col(Configuration[1](index_ket_n,jj));
            }


            Basis.AMP(bra,ket,J,Parity,h_v,n_v);

            int count_num = 0;
            for(int loop=0;loop<J_size;loop++)
            {
                const int bs = blocksize(loop);
                const int offset_i = i*bs;
                const int offset_j = j*bs;

                cx_mat h_sub(h_v.subvec(count_num,count_num+bs*bs-1));
                cx_mat n_sub(n_v.subvec(count_num,count_num+bs*bs-1));
                h_sub.reshape(bs,bs);
                n_sub.reshape(bs,bs);

                H[loop](offset_i,offset_j,size(bs,bs)) = h_sub;
                N[loop](offset_i,offset_j,size(bs,bs)) = n_sub;

                count_num += bs*bs;
            }
        }
    }


    ofstream ofile("Spectra.txt");
    ofile<<"The spectral of nucleus Z = "<<Basis.MassNumber[0]<<", A = "<<Basis.MassNumber[iso]<<endl;
    cout<<"The spectral of nucleus Z = "<<Basis.MassNumber[0]<<", A = "<<Basis.MassNumber[iso]<<endl;
    for(int loop=J.n_elem-1;loop>=0;loop--)
    {
        for(int i=H[loop].n_rows-1;i>=0;i--)
        {
            for(int j=i-1;j>=0;j--)
            {
                H[loop](j,i)=conj(H[loop](i,j));
                N[loop](j,i)=conj(N[loop](i,j));
            }
        }

        Basis.HW_Equation_Hermitian(H[loop],N[loop],E[loop],wf[loop]);

        ofile<<"J: "<<J(loop)<<", P: "<<Parity(loop)<<endl;
        for(int i = min(Levels_show,E[loop].n_elem)-1;i>=0;i--)
        {
            ofile<<i<<" "<<E[loop](i)<<endl;
            cout<<i<<", J: "<<J(loop)<<", P: "<<Parity(loop)<<", Energy: "<<E[loop](i)<<" MeV"<<endl;
        }
    }

    ofile.close();
}

void Nucleus::Spectra ()
/** \brief Spectra without angular momentum projection
 *
 * \return Excited energies
 *
 */
{
    const unsigned int Particle_Hole_Truncation = 0;
    double Total_Cutoff = cutoff[0];

    const int iso = Interaction<double>::IsoSpin_Z;
    const int parity_dim = HartreeFock::Parity_Decoupling?Interaction<double>::Parity_Domain:1;
    const unsigned int Levels_show = 12;
    const double jmin = Basis.MassNumber[iso]&1?0.5:0;
    ///!!!!!!double cutoff[iso] = {30,0};

    vec E_cutoff(cutoff,iso);

    Basis.Iteration(Basis.ME);
	Basis.Set_HF_Basis();
	Basis.Save_HF_Basis("hfsps");
//	for(int i=0; i<=Basis.bra[0].n_rows-1; i++)
//	{
//		for(int j=0; j<=Basis.bra[0].n_cols-1;j++)
//		cout << Basis.bra[0](i,j).real() << "  ";
//		cout << endl;
//	}

    ///!!!Basis.Iteration_BHF(Basis.ME);
    ivec dim_max = Construct_Configuration(E_cutoff,Particle_Hole_Truncation);

    umat Config[iso];

    cx_mat bra_sub[iso],ket_sub[iso];
    for(int i=iso-1;i>=0;i--)
    {
        if(Configuration[i].is_empty()) continue;

        Config[i] = zeros<umat>(Basis.MassNumber[i],Configuration[i].n_rows);
        Config[i].rows(Basis.MassNumber[i]-Configuration[i].n_cols,Basis.MassNumber[i]-1) = Configuration[i].t();
        for(int n=Basis.MassNumber[i]-Configuration[i].n_cols-1;n>=0;n--)
        {
            for(int m=Configuration[i].n_rows-1;m>=0;m--) Config[i](n,m) = n;
        }

        if((Basis.bra[i].n_cols>0)&&(Basis.ket[i].n_cols>0)&&(dim_max(i)>0))
        {
            bra_sub[i] = Basis.bra[i].cols(0,dim_max(i)-1);
            ket_sub[i] = Basis.ket[i].cols(0,dim_max(i)-1);
        }
//		for(int k=0; k<=bra_sub[i].n_rows-1; k++)
//		{
//			for(int j=0; j<=bra_sub[i].n_cols-1; j++)
//			  cout << bra_sub[i](k,j).real() << "  ";
//			cout << endl;
//		}
    }

    Basis.ME_HF.Basis_Transform(Basis.ME,bra_sub,ket_sub,Configuration_M);
	Basis.ME_HF.ME_Output_Mscheme();

    const int size1 = max(int(Configuration[0].n_rows),1);
    const int size2 = max(int(Configuration[1].n_rows),1);

    imat Combination[parity_dim];
    imat Combination_duplicate[parity_dim];
    int dim[parity_dim] = {0};

    for(int loop=parity_dim-1;loop>=0;loop--)
    {
        Combination_duplicate[loop] = zeros<imat>(size1*size2,iso);
        dim[loop] = 0;
    }

    ivec parity[iso];
    if(HartreeFock::Parity_Decoupling)
    {
        for(int loop=iso-1;loop>=0;loop--)
        {
            parity[loop] = zeros<ivec>(Config[loop].n_cols);

            for(int i=Config[loop].n_cols-1;i>=0;i--)
            {
                parity[loop](i) = int(prod(Basis.P_record[loop](Config[loop].col(i)))+parity_dim+0.5)-parity_dim;
            }
        }
    }

    cout<<"Total energy truncation is "<<Total_Cutoff<<" MeV."<<endl;
    for(int i=size1-1;i>=0;i--)
    {
        int p1 = parity[0].is_empty()?1:parity[0](i);
        double E1 = Excitation_E[0].is_empty()?0:Excitation_E[0](i);

        for(int j=size2-1;j>=0;j--)
        {
            int p2 = parity[1].is_empty()?1:parity[1](j);
            int index = HartreeFock::Parity_Decoupling?((1-p1*p2)>>1):0;
            double E2 = Excitation_E[1].is_empty()?0:Excitation_E[1](j);

//            if((E1+E2)>Total_Cutoff) continue;

            if(HartreeFock::Axial_Symmetry)
            {
                double temp = Configuration[0].is_empty()?0:Configuration_M[0](i);
                temp += Configuration[1].is_empty()?0:Configuration_M[1](j);
                if(int(fabs(temp-jmin)+0.5)==0)
                {
                    if(Particle_Hole_Truncation)
                    {
                        uvec q1,q2;
                        if(!Configuration[0].is_empty()) q1=find(Configuration[0].row(i)>=Basis.MassNumber[0]);
                        if(!Configuration[1].is_empty()) q2=find(Configuration[1].row(j)>=Basis.MassNumber[1]);

                        if((q1.n_elem+q2.n_elem)<=Particle_Hole_Truncation)
                        {
                            Combination_duplicate[index](dim[index],0) = i;
                            Combination_duplicate[index](dim[index]++,1) = j;
                        }
                    }
                    else
                    {
                        Combination_duplicate[index](dim[index],0) = i;
                        Combination_duplicate[index](dim[index]++,1) = j;
                    }
                }
            }
            else
            {
                if(Particle_Hole_Truncation)
                {
                    uvec q1,q2;
                    if(!Configuration[0].is_empty()) q1=find(Configuration[0].row(i)>=Basis.MassNumber[0]);
                    if(!Configuration[1].is_empty()) q2=find(Configuration[1].row(j)>=Basis.MassNumber[1]);

                    if((q1.n_elem+q2.n_elem)<=Particle_Hole_Truncation)
                    {
                        Combination_duplicate[index](dim[index],0) = i;
                        Combination_duplicate[index](dim[index]++,1) = j;
                    }
                }
                else
                {
                    Combination_duplicate[index](dim[index],0) = i;
                    Combination_duplicate[index](dim[index]++,1) = j;
                }
            }
        }
    }
    if(HartreeFock::Parity_Decoupling)
    {
        cout<<"Total dimension for parity \'+\' is "<<dim[0]<<endl;
        if(dim[0]>0) Combination[0] = Combination_duplicate[0].submat(0,0,dim[0]-1,iso-1);
        cout<<"Total dimension for parity \'-\' is "<<dim[1]<<endl;
        if(dim[1]>0) Combination[1] = Combination_duplicate[1].submat(0,0,dim[1]-1,iso-1);
    }
    else
    {
        cout<<"Total dimension is "<<dim[0]<<endl;
        if(dim[0]>0) Combination[0] = Combination_duplicate[0].submat(0,0,dim[0]-1,iso-1);
    }

    const int nthreads = omp_get_num_procs();
    omp_set_num_threads(nthreads);

    vector<bitset<Interaction<double>::Max_Bit> > psi[iso];
    for(int loop=iso-1;loop>=0;loop--)
    {
        psi[loop].resize(Config[loop].n_cols);

        #pragma omp parallel for
        for(int i=Config[loop].n_cols-1;i>=0;i--)
        {
            for(int j=Config[loop].n_rows-1;j>=0;j--) psi[loop][i].set(Config[loop](j,i));
        }
    }

    cx_mat H[parity_dim];

    for(int loop=parity_dim-1;loop>=0;loop--)
    {
        if(dim[loop]>0) H[loop] = zeros<cx_mat>(dim[loop],dim[loop]);

        const int com_size = Combination[loop].n_rows;

        #pragma omp parallel for
        for(int i=com_size-1;i>=0;i--)
        {
            bitset<Interaction<double>::Max_Bit> psil[iso];
            bitset<Interaction<double>::Max_Bit> psir[iso];

            for(int j=iso-1;j>=0;j--)
            {
                if(!psi[j].empty()) psil[j] = psi[j][Combination[loop](i,j)];
            }

            for(int j=i;j>=0;j--)
            {
                for(int k=iso-1;k>=0;k--)
                {
                    if(!psi[k].empty()) psir[k] = psi[k][Combination[loop](j,k)];
                }

                H[loop](i,j) = Basis.Construct_ME(Basis.ME_HF,psil,psir);
            }
        }

        H[loop] = symmatl(H[loop]);
    }

    ofstream ofile("Spectra.txt");
    ofile<<"The spectra of nucleus Z = "<<Basis.MassNumber[0]<<", A = "<<Basis.MassNumber[iso]<<endl;
    cout<<"The spectra of nucleus Z = "<<Basis.MassNumber[0]<<", A = "<<Basis.MassNumber[iso]<<endl;
    vec E[parity_dim];

//	cout << H[0](0,0) << endl;
    #pragma omp parallel for
    for(int loop=parity_dim-1;loop>=0;loop--)
    {
/*		for(int i=0;i<H[loop].n_cols;i++)
		{
//			cout << H[loop](i,i);
			for(int j=0;j<H[loop].n_cols;j++)
//				if(abs(H[loop](i,j).imag())>0.01)
				{
//					cout << "Complex matrix." <<endl;
//					cout << H[loop](i,j);
				}
//			cout << endl;
		}*/
		cout << "Diagonalization start." << endl;
		if(H[loop].n_cols>300)
		{
			sp_mat Hsp(real(H[loop]));
			eigs_sym(E[loop],Hsp,12,"sa");
		}
		else
		{
			eig_sym(E[loop],H[loop]);
		}
    }

    for(int loop=parity_dim-1;loop>=0;loop--)
    {
        ofile<<"Parity "<<loop<<endl;
        cout<<"Parity "<<loop<<endl;
        for(int i = min(Levels_show,E[loop].n_elem)-1;i>=0;i--)
        {
            ofile<<i<<" "<<E[loop](i)<<" MeV"<<endl;
            cout<<i<<", Energy: "<<E[loop](i)<<" MeV"<<endl;
        }
    }

    ofile.close();
}

void Nucleus::Spectra_GCM ()
/** \brief Spectra with angular momentum projection
 *
 * \return Excited energies
 *
 */
{
    ///#pragma omp parallel for
    for(int i=-10;i<=19;i++)
    {
        HartreeFock Basis_GCM(Basis.MassNumber[0],Basis.MassNumber[2],Basis.ME.Dimension_M);
        Basis_GCM.Iteration(Basis.ME,double(i));
    }
}

/***********************************************************************************************************************/

int main(int argc, char** argv)
{
    const int nn =5;
    vec J = zeros<vec>(nn);
    ivec P = zeros<ivec>(nn);

    int Z(6),A(12);
    cout<<"Choose the nucleus you want to calculate."<<endl;
    cout<<"Enter the proton number and mass number, respectively."<<endl;
    ifstream input("input.dat");
	input>>Z;
	input>>A;
	cout << "  " <<Z << "  " << A << endl;

    for(int i=nn-1;i>=0;i--)
    {
        if(A&1) J(i)=i+0.5;
        else J(i)=i;

        P(i)=1;
    }

    cout<<"Enter the energy truncations for proton and neutron, respectively."<<endl;
    input>>cutoff[0];
	input>>cutoff[1];
	cout << "  " << cutoff[0] << "  " << cutoff[1] << endl;

    wall_clock timer;
    timer.tic();
//	ifstream spint("spint.dat");
	string sp, interaction;
	input >> sp;
	input >> interaction;
//	spint.close();
	double beta;
	input >> beta;
//	double beta = 10.;
    Nucleus SD(Z,A,sp,interaction,beta);
    ///Nucleus SD(Z,A,"sd.sps","usdb.int");
    ///Nucleus SD(Z,A,"sps4.dat","vint4.dat");

    cout<<"Need angular momentum projection?"<<endl;
/*    SD.Spectra_GCM();*/

    bool projection = false;
    input>>projection;
	cout << "  " << projection << endl;
	input.close();
    if(projection)
    {
        //SD.Spectra_MemOptimize(J,P);
        SD.Spectra(J,P);
    }
    else
    {
        SD.Spectra();
    }

    cout<<"During time is "<<timer.toc()<<" seconds."<<endl;

/*
    HartreeFock Mg24(4,8,"sd.sps","Yoooooooooo");
    Mg24.Iteration();
    Mg24.DBP();*/

    return 0;
}
