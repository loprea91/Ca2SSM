!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 
!   Ca2+ transient model (Dejardins et al., 2022)
!---------------------------------------------------------------------- 
!----------------------------------------------------------------------
!

 SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!--------- ---- 

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
    DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
    DOUBLE PRECISION, INTENT(OUT) :: F(NDIM), DFDU(NDIM,*), DFDP(NDIM,*)
    
    ! Constants
    DOUBLE PRECISION, PARAMETER :: vncx=1.4, eta=0.35, Va=-80., P=96.5, Ra=8.314, temp=300. 
    DOUBLE PRECISION, PARAMETER :: nao=14., ksat=0.25, kmcao=1.3, kmnao=97.63, kmnai=12.3 
    DOUBLE PRECISION, PARAMETER :: kmcai=0.0026, kn=0.1, cao=12.
    DOUBLE PRECISION, PARAMETER :: ccai=0.5, v3=120., hc3=2., k3=0.3, v2=0.5
    DOUBLE PRECISION, PARAMETER :: ip3=0.2, d5=0.08234
    DOUBLE PRECISION, PARAMETER :: k_pmca=0.8, v_pmca=0.6, kb1=0.2573
    DOUBLE PRECISION, PARAMETER :: fi=0.01, gammaa=9., fe=0.025
    DOUBLE PRECISION, PARAMETER :: a2=0.2, d1=0.13, d2=1.049, d3=0.9434
    DOUBLE PRECISION, PARAMETER :: Ks=50., tau_soc=30., ka=0.0192, kb=0.2573, kc=0.0571
    DOUBLE PRECISION, PARAMETER :: kca=5., kna=5., tau_o=10., ktau=1.

    ! Variables
    DOUBLE PRECISION :: Ca, Cer, h, soc, w, x, Na
    ! Bifurcation Parameters
    DOUBLE PRECISION :: Vs, v_ip3, vr
    DOUBLE PRECISION :: hinf, minf, ninf, ninfi, Kx, VFRT, soc_inf, winf, xinf
    DOUBLE PRECISION :: Jip3, Jserca, Jleak, Jryr, Jsoce, Jncx, Jpmca 
    ! Definition of variables
    Ca = U(1)
    Cer = U(2)
    h = U(3)
    soc = U(4)
    w = U(5)
    x = U(6)
    Na  = U(7)

    ! Definition of parameters
    Vs = PAR(1)
    v_ip3 = PAR(2)
    vr = PAR(3)

    ! Equations of the system
    hinf=(d2*(ip3+d1)/(ip3+d3))/((d2*(ip3+d1)/(ip3+d3))+Ca)
    minf=ip3/(ip3 + d1)
    ninf=Ca/(Ca+d5)
    ninfi=1/(1+(kn/Ca)**2)
    Kx=kmcao*(Na**3)+(kmnao**3)*Ca+(kmnai**3)*cao*(1+Ca/kmcai)+kmcai*(nao**3)*(1+(Na**3)/(kmnai**3))+(Na**3)*cao+(nao**3)*Ca
    VFRT=Va*P/(Ra*temp)
    soc_inf=(Ks**4)/((Ks**4)+(Cer**4))   
    winf=(ka/(Ca**4)+1+(Ca**3)/kb)/(1/kc+ka/(Ca**4)+1+(Ca**3)/kb)
    xinf=1-1/((1+(Ca/kca)**2)*(1+(kna/Na)**2))
    Jip3 = v_ip3*(minf**3)*(ninf**3)*(h**3)*(Cer-Ca)
    Jserca = v3*(Ca**hc3)/((k3**hc3)+(Ca**hc3))
    Jleak = v2*(Cer-Ca)
    Jryr = vr*w*(1+(Ca**3)/kb1)/(ka/(Ca**4)+1+(Ca**3)/kb1)*(Cer-Ca)
    Jsoce = Vs*soc
    Jncx = ninfi*x*(vncx*(exp(eta*VFRT)*(Na**3)*cao-exp((eta-1)*VFRT)*(nao**3)*Ca)/(Kx*(1+ksat*(exp((eta-1)*VFRT)))))
    Jpmca = (v_pmca*(Ca**2)/((Ca**2)+(k_pmca**2)))

    ! ODEs
    F(1)=fi*(Jip3-Jserca+Jleak+Jryr+Jsoce+Jncx-Jpmca)
    F(2)=-gammaa*fe*(Jip3-Jserca+Jleak+Jryr)
    F(3)=(hinf-h)/(1/(a2*((d2*(ip3+d1)/(ip3+d3))+Ca)))
    F(4)=(soc_inf-soc)/tau_soc
    F(5)=(winf-w)/(winf/kc)
    F(6)=(xinf-x)/(0.25+tau_o/(1+(Ca/ktau)))
    F(7)=-3*Jncx

 END SUBROUTINE FUNC

!---------------------------------------------------------------------- 
 SUBROUTINE STPNT(NDIM,U,PAR,T)
!--------- ---- 

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM
    DOUBLE PRECISION, INTENT(IN) :: T
    DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
    
    ! Initial conditions
    U(1:7)=(/  0.28987, 40.632, 0.51087, 0.69633, 0.17883, 0.1652, 11.356 /)
    PAR(1:3)=(/ 0.1, 0.88, 18. /)
 END SUBROUTINE STPNT

!---------------------------------------------------------------------- 
SUBROUTINE PVLS(NDIM,U,PAR)
!     ---------------
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

      DOUBLE PRECISION, EXTERNAL :: GETP
      INTEGER NDX,NCOL,NTST

!     ---------------
!---------------------------------------------------------------------- 

!  Set PAR(3) equal to the minimum of U(2)
       PAR(4)=GETP('MIN',1,U)
! The following subroutines are not used here,
! but they must be supplied as dummy routines
      END SUBROUTINE PVLS
      SUBROUTINE BCND 
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT



!----------------------------------------------------------------------
!----------------------------------------------------------------------
!---------------------------------------------------------------------- 
! END SUBROUTINE PVLS

!---------------------------------------------------------------------- 
! SUBROUTINE BCND
! END SUBROUTINE BCND

!---------------------------------------------------------------------- 
! SUBROUTINE ICND 
! END SUBROUTINE ICND

!----------------------------------------------------------------------
! SUBROUTINE FOPT 
! END SUBROUTINE FOPT

