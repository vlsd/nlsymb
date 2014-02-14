#ifndef ADJOINT_HPP
#define ADJOINT_HPP

//[ The rhs of rho' defined as a class
// USER SPECIFIED:
class adjoint {
  state_intp & m_rx_intp;
  double m_g, indx;
  state_type m_x, m_rho;
  state_type m_u1;
  sys_lin m_lin;
  Eigen::Matrix< double, xlen, 1 > m_mx;
  Eigen::Matrix< double, xlen, 1 > m_mrho;
  Eigen::Matrix< double, xlen, 1 > m_mrhodot;
  Eigen::Matrix< double, xlen, xlen > m_mdfdx;
  
public:
  adjoint( state_intp & x_intp ) :  m_g(9.81) , m_rx_intp( x_intp ) , 
				    m_u1(ulen), m_x(xlen),
				    m_rho(xlen)/*, m_lin( )*/ {  
    for ( size_t i=0; i<ulen; i++ ) { m_u1[i] = 0.0; } 
  }

  void operator() (const state_type &rho, state_type &rhodot, const double t)
  {
    m_rho = rho;
    m_rx_intp(t, m_x);        // store the current state in x
    State2Mat( m_x, m_mx );   // convert state to matrix form
    State2Mat( m_rho, m_mrho );
    //
    m_lin.A( m_x, m_u1, m_mdfdx );
    //
    AngleWrap( m_mx[0] ); // Only for angle wrapping
    AngleWrap( m_mx[2] );
    //
    m_mrhodot = -Q*m_mx - m_mdfdx.transpose()*m_mrho;
    //
    for (indx = 0; indx < xlen; indx++ ) { rhodot[indx] = m_mrhodot[indx]; }
  }
};
//]

#endif  // ADJOINT_HPP
