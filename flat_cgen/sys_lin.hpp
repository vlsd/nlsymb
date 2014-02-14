#ifndef SYS_LIN_HPP
#define SYS_LIN_HPP

//[ Linearizations of the system defined as a class
// USER SPECIFIED:
class sys_lin {

public:
  sys_lin( ) : {  }

  void A( const state_type & x, const state_type & u,
          Eigen::Matrix< double, xlen, xlen > & Amat ) {
    Amat = 0*Amat;
    Amat(0,2) = 1;
    Amat(1,3) = 1; 
  }

  void B( const state_type & x, const state_type & u,
          Eigen::Matrix< double, xlen, ulen > & Bmat ) {
    Bmat = Bmat*0;
    Bmat(2,0) = 1;
    if (x[1] > 0)
        Bmat(3,1) = 1;
    else
        Bmat(3,1) = -1;
  }
};
//]

#endif  // SYS_LIN_HPP
