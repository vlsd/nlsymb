#ifndef SYS_DYNAM_HPP
#define SYS_DYNAM_HPP

//[ The rhs of xdot = f(x) defined as a class
// USER SPECIFIED:
class sys_dynam {
  b_control & m_u;
  state_type u;

public:
  sys_dynam( b_control & uu ) : m_u(uu) , u(ulen) {  }

  void operator() (const state_type &x, state_type &dxdt, const double t)
  {
    const int si=1; // special index
    m_u(t, u);
    //
    dxdt[0] = x[2];
    dxdt[1] = x[3];
    dxdt[2] = u[0];
    if ( x[si] > 0 ) 
        dxdt[3] = u[1] - 9.8;
    else
        dxdt[3] = -u[1] + 9.8;
  }
};
//]

#endif  // SYS_DYNAM_HPP
