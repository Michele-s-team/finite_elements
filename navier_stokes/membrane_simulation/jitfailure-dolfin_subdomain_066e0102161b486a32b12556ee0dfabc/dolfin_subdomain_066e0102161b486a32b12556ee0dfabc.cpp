
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/common/Array.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/SubDomain.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_subdomain_066e0102161b486a32b12556ee0dfabc : public SubDomain
  {
     public:
       

       dolfin_subdomain_066e0102161b486a32b12556ee0dfabc()
          {
            
          }

       // Return true for points inside the sub domain
       bool inside(const Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const final
       {
         return on_boundary && ((x[0]-c_r[0])*(x[0]-c_r[0]) + (x[1]-c_r[1])*(x[1]-c_r[1]) < (0.2*0.2));
       }

       void set_property(std::string name, double value)
       {

       }

       double get_property(std::string name) const
       {

         return 0.0;
       }

  };
}

extern "C" DLL_EXPORT dolfin::SubDomain * create_dolfin_subdomain_066e0102161b486a32b12556ee0dfabc()
{
  return new dolfin::dolfin_subdomain_066e0102161b486a32b12556ee0dfabc;
}

