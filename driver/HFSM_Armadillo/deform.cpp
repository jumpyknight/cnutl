#include <iostream>
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_coupling.h>
#include <armadillo>
using namespace std;
using namespace arma;

template <typename T>
class Interaction;
class HartreeFock;

template <typename T>
void HartreeFock :: deform(cx_double& dbeta,cx_double& dgamma,cx_mat *rho,int iso,Interaction<T> & Force)
{
    double eps,pi;
    unsigned int n_states;
    int i,j,m;
    double l1,ij1,j1,n1,m1,im1,sign,l2,n2,ij2,j2,m2,delta,f1,f2,f3,f4,f5,f6;
    cx_mat q(3,3),qp(3,3),qn(3,3);
    cx_double q1, q2, q_diag[4];
    cx_double trace,r2trace;
    eps = 1.E-4;
    pi = acos(-1.);
    n_states=Force.Dimension_M+1
    double **r2=new double* [n_states];
    for (i=0;i<=n_states;i++)
        r2[i] = new double [nstates];
    double ***q0=new double** [n_states];
    for (i=0;i<=n_states;i++)
        {
        q0[i] = new double* [nstates];
        for (j=0;j<=nstates;j++)
            q0[i][j] = new double [7];
        }
//    new(r2(n_states,n_states),q0(n_states,n_states,6))

      for (i=1;i<=n_states;i++)
      {
        l1=Force.M_scheme[0][i-1].l;
        ij1=Force.M_scheme[0][i-1].j*2.;
        j1=ij1/2.0;;
        n1=Force.M_scheme[0][i-1].n;
        m1=Force.M_scheme[0][i-1].m;
        im1=m1*2.;
        sign=(-1)**((ij1-im1)/2);
        for (j=1;j<=n_states;j++)
        {
          l2=Force.M_scheme[0][j-1].l;
          n2=Force.M_scheme[0][j-1].n;
          ij2=Force.M_scheme[0][j-1].j*2.;
          j2=ij2/2.0;
          m2=Force.M_scheme[0][j-1].m;

          delta=0.0;
          if((m1==m2) && (ij1==ij2) && (l1==l2))delta=1.0;

          f1=sign*gsl_sf_coupling_3j(2.*j1,4.0,2.*j2,-2.*m1,-4.,2.*m2);
          f1=f1*YredME(2,l1,ij1,l2,ij2)/2.;

          f2=sign*gsl_sf_coupling_3j(2.*j1,4.0,2.*j2,-2.*m1,-2.,2.*m2);
          f2=f2*YredME(2,l1,ij1,l2,ij2)/2.;

          f3=sign*gsl_sf_coupling_3j(2.*j1,4.0,2.*j2,-2*m1, 0.,2.*m2);
          f3=f3*YredME(2,l1,ij1,l2,ij2)/2.;

          f4=sign*gsl_sf_coupling_3j(2.*j1,4.0,2.*j2,-2.*m1,+2.,2.*m2);
          f4=f4*YredME(2,l1,ij1,l2,ij2)/2.;

          f5=sign*gsl_sf_coupling_3j(2.*j1,4.0,2.*j2,-2.*m1,+4.,2.*m2);
          f5=f5*YredME(2,l1,ij1,l2,ij2)/2.;

          f6=HORadInt(2,n1,l1,n2,l2);
          r2[i][j]=f6*delta;

          q0[i][j][1]= f6*((f1+f5)*sqrt(6./5.)-2.*f3/sqrt(5.));        /// xx
          q0[i][j][2]=-f6*((f1+f5)*sqrt(6./5.)+2.*f3/sqrt(5.));        /// yy
          q0[i][j][3]= 4.*f6*f3/sqrt(5.);                              /// zz
          q0[i][j][4]= f6*sqrt(6./5.)*(f1-f5); /// *I (purely imaginary)// xy
          q0[i][j][5]=-f6*sqrt(6./5.)*(f4-f2);                         /// xz
          q0[i][j][6]= f6*sqrt(6./5.)*(f4+f2); /// *I (purely imaginary)// yz
        }
      }
//C      stop
        int **map2=new int *[7];
        for (i=0;i<=6;i++)
            map2[i]=new int [3];
      for (m=1;m<=3;m++)
      {
        map2[m][1]=m;
        map2[m][2]=m;
      }
      map2[4][1]=1;
      map2[4][2]=2;
      map2[5][1]=1;
      map2[5][2]=3;
      map2[6][1]=2;
      map2[6][2]=3;

      double **tmp=new *[n_states];
      for(i=0;i<=n_states;i++)
        tmp[i]=new [n_states];

    if(iso!=2)
    {
        cout << "iso!=2, code needs updating."
        return;
    }
    cx_mat rhop=rho[0];
    cx_mat rhon=rho[1];

//      allocate(tmp(n_states,n_states))
      for (m=1;m<=6;m++)
      {
        for (i=1;i<=n_states;i++)
        for (j=1;j<=n_states;j++)
             tmp[i][j]=q0[i][j][m];
         trace = compute_trace(tmp,rhop,n_states);
         qp(map2[m][1]-1,map2[m][2]-1)=trace;
         qp(map2[m][2]-1,map2[m][1]-1)=trace;
         trace = compute_trace(tmp,rhon,n_states);
         qn(map2[m][1]-1,map2[m][2]-1)=trace;
         qn(map2[m][2]-1,map2[m][1]-1)=trace;
      }

      trace = compute_trace(r2,rhop,n_states);
      r2trace = compute_trace(r2,rhon,n_states);
      r2trace = r2trace+trace;  //+r20

//C Set the imaginary elements to zero

      if(abs(qp(0,1)>eps || abs(qp(1,2)>eps ||abs(qn(0,1)>eps ||abs(qn(1,2)>eps)
        {
            cout << qp(0,1) << "  " << qp(1,2) << "  " << qn(0,1) << "  " << qn(1,2) << endl;
          cout << "Imaginary ME for Q" << endl;
          return;
        }

      qp(0,1)=0.0;
      qp(1,0)=0.0;
      qp(1,2)=0.0;
      qp(2,1)=0.0;

      qn(0,1)=0.0;
      qn(1,0)=0.0;
      qn(1,2)=0.0;
      qn(2,1)=0.0;

      for (i=0;i<=2;i++)
         for (j=0;j<=2;j++)
            q(i,j)=qp(i,j)+qn(i,j);
//      do i=1,3
//          write(82,192)(q(i,j),j=1,3)
//      enddo
//      write(82,*)
//192   format(3(F12.6,1X))
//      call eigval(q,3,3,q_diag,vecq,work)
//      call eigsrt(q_diag,vecq,3,3)

       vec q_eigval(3)=eig_sym(q);
        q_diag[1]=q_eigval(0);
        q_diag[2]=q_eigval(1);
        q_diag[3]=q_eigval(2);

       intrinsic(q_diag,q1,q2);

      dgamma=atan(q2*sqrt(2.)/q1)*180./pi;
      if(dgamma<0.0)dgamma=-dgamma;
//      beta=q1*cos(gamma)+q2*sqrt(2.)*sin(gamma)
//      beta=sqrt(5.*pi)*beta/3.

      dbeta=0.0;
      for (i=1;i<=3;i++)
         dbeta=dbeta+q_diag(i)**2;
      dbeta=5.*sqrt(dbeta/6.)/6.;  // 5/8/2002 Why?
//      beta=5.*sqrt(beta)/4./pi   ! The original factor in Eq. 27
//                                 ! (PRC 49, 1442) has been changed to
//                                 ! 5/(4*Pi) to account for core nucleons

//      beta=beta*2*pi*sqrt(5./4./Pi)/3.
//      beta=4*Pi*beta/5.
      dbeta=dbeta/r2trace;

      return;
}






      double YredME(int L,int l1,int j1,int l2,int j2)
/*C==================================================================
C Calculates the reduced matrix element
C              <(l1 1/2) j1 || Y_L || (l2 1/2) j2> * sqrt(4*pi)
C==================================================================
*/
//      integer j1,j2
//      integer L,l1,l2
{
      double xj1,xj2;

      double fact,ll1,ll2,LL;
        double yredme;
      LL=double(L);
      ll1=double(l1);
      ll2=double(l2);
      xj1=j1/2.;
      xj2=j2/2.;

      fact=(L+(j2+1)/2)&1?-1:1;
      fact=fact*sqrt((2.*ll1+1)*(2.*ll2+1.)*(2.*LL+1.)*(j1+1)*(j2+1));

      yredme=gsl_sf_coupling_6j(2.*ll1,2.*xj1,1.,2.*xj2,2.*ll2,2.*LL)*gsl_sf_coupling_3j(2.*ll1,2.*LL,2.*ll2,0.0,0.0,0.0);
      yredme=fact*yredme;

//C      write(*,*)sixj(ll1,j1,0.5,xj2,ll2,LL)
      return yredme;
}





      double HORadInt(int L,int n1,int l1,int n2,int l2)
/*C
C   computes radial integral R(n1,l1, alpha*r) R(n2,l2, alpha*r) r^(L+2)
C   where R's are h.o. radial functions
C
C   uses Lawson 1.11a; alpha = 1  must scale by 1/alpha^2 to get final
C   answer
C*/
//      implicit none

//      integer L			! weighting of r ( = L in Ylm)
//      integer n1,n2,l1,l2       ! q#'s of states

 {
        double sum;			// result
      int q;			// dummy for summation
      int qmin,qmax;		// limits of summation

      double lnprefact,lnsum,horadint;

//C............let's get going..................
      if( (l1+l2+L)%2 != 0) 	  // must be even overall
	{horadint = 0.0  ;
	return horadint;}

/*C............. l1,l2,L must satisfy triangle relation....
C      if( ( l1+l2 .lt. L) .or. (abs(l1-l2) .gt. L) )then
C	HORadInt = 0.0
C	return
C      endif*/
      lnprefact = (gsl_sf_lnfact(n1)+gsl_sf_lnfact(n2) + log(2.)*(n1+n2-L)- gsl_sf_lndoublefact(2*n1+2*l1+1) - gsl_sf_lndoublefact(2*n2+2*l2+1))/2. + gsl_sf_lnfact( (l2-l1+L)/2) + gsl_sf_lnfact( (l1-l2+L)/2);

      qmax = min(n1,n2);
      qmin = max(0,max( n1-(l2-l1+L)/2,n2-(l1-l2+L)/2));
      sum = 0.0;
      for (q = qmin;q<=qmax;q++)
	{
	    lnsum =  gsl_sf_lndoublefact(l1+l2+L+2*q+1) -q*log(2.) - gsl_sf_lnfact(q) - gsl_sf_lnfact(n1-q) -gsl_sf_lnfact(n2-q) - gsl_sf_lnfact(q+(l1-l2+L)/2-n2) - gsl_sf_lnfact(q+(l2-l1+L)/2-n1);

        sum = sum + exp( lnprefact+lnsum);
//C       write(20,*)exp( lnprefact+lnsum)
	}

      if(  abs(n1-n2)%2 !=0) sum = -sum;

//C      write(*,*)L,n1,l1,n2,l2,sum
      horadint = sum;
      return horadint;
}





      void intrinsic(cx_double* q_diag,cx_double& q1,cx_double& q2)
/*C===================================================================
      implicit none
C===================================================================
C Calculates the intrinsic values for the quadrupole operator
C as defined in PRC 61, 034303 (2000)
C===================================================================
*/
//      real q_diag(3),q1,q2
{
      double pi;

      pi=acos(-1.);

      q1=q_diag[3]*sqrt(5./pi)/4.;

      q2=sqrt(5./(2.*pi))*q_diag[1]+q1*sqrt(2.);
      q2=q2/(2.*sqrt(3.));

      return;
}




      cx_double compute_trace(double ** a,cx_mat rho,int n_states)
/*C====================================================================
C Calculates trace=Tr(a*rho)
C===================================================================
      implicit none
C==========================================================================
C Subroutines Called: none
C Called by: the main program
C
C INPUT:
C
C OUTPUT:
C
C==========================================================================
*/
/*      integer,intent(IN)     :: n_states
      real,intent(IN)        :: a(n_states,n_states)
      real*8,intent(IN)      :: rho(n_states,n_states)

      real,intent(OUT)       :: trace
*/
{
      int i,j;
      cx_double sum,trace;

      sum=0.0;
      for (i=1;i<=n_states;i++)
         for (j=1;j<=n_states;j++)
	         sum=sum+rho(i-1,j-1)*a[j][i];

      trace=sum;

      return trace;
}
