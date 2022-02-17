#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
#include <boost/tuple/tuple_comparison.hpp>
#include <limits>
#include <map>

namespace bp = boost::python;
namespace ei = Eigen;
namespace bpn = boost::python::numpy;

typedef ei::Matrix<float, 3, 3> Matrix3f;
typedef ei::Matrix<float, 3, 1> Vector3f;

typedef boost::tuple< std::vector< std::vector<float> >, std::vector< std::vector<uint8_t> >, std::vector< std::vector<uint32_t> >, std::vector<std::vector<uint32_t> > > Custom_tuple;
typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t> > Components_tuple;
typedef boost::tuple< std::vector<uint8_t>, std::vector<uint8_t> > Subgraph_tuple;

typedef boost::tuple< uint32_t, uint32_t, uint32_t > Space_tuple;

struct VecToArray
{//converts a vector<uint8_t> to a numpy array
    static PyObject * convert(const std::vector<uint8_t> & vec) {
    npy_intp dims = vec.size();
    PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT8);
    void * arr_data = PyArray_DATA((PyArrayObject*)obj);
    memcpy(arr_data, &vec[0], dims * sizeof(uint8_t));
    return obj;
    }
};

template <class T>
struct VecvecToArray
{//converts a vector< vector<uint32_t> > to a numpy 2d array
    static PyObject * convert(const std::vector< std::vector<T> > & vecvec)
    {
        npy_intp dims[2];
        dims[0] = vecvec.size();
        dims[1] = vecvec[0].size();
        PyObject * obj;
        if (typeid(T) == typeid(uint8_t))
            obj = PyArray_SimpleNew(2, dims, NPY_UINT8);
        else if (typeid(T) == typeid(float))
            obj = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        else if (typeid(T) == typeid(uint32_t))
            obj = PyArray_SimpleNew(2, dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        std::size_t cell_size = sizeof(T);
        for (std::size_t i = 0; i < dims[0]; i++)
        {
            memcpy(arr_data + i * dims[1] * cell_size, &(vecvec[i][0]), dims[1] * cell_size);
        }
        return obj;
    }
};

struct VecToArray32
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};


template<class T>
struct VecvecToList
{//converts a vector< vector<T> > to a list
        static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};

struct to_py_tuple
{//converts to a python tuple
    static PyObject* convert(const Custom_tuple & c_tuple){
        bp::list values;

        PyObject * pyo1 = VecvecToArray<float>::convert(c_tuple.get<0>());
        PyObject * pyo2 = VecvecToArray<uint8_t>::convert(c_tuple.get<1>());
        PyObject * pyo3 = VecvecToArray<uint32_t>::convert(c_tuple.get<2>());
        PyObject * pyo4 = VecvecToArray<uint32_t>::convert(c_tuple.get<3>());

        values.append(bp::handle<>(bp::borrowed(pyo1)));
        values.append(bp::handle<>(bp::borrowed(pyo2)));
        values.append(bp::handle<>(bp::borrowed(pyo3)));
        values.append(bp::handle<>(bp::borrowed(pyo4)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};


PyObject * compute_norm(const bpn::ndarray & xyz ,const bpn::ndarray & target, int k_nn)
{//compute the following geometric features (geof) features of a point cloud:
 //linearity planarity scattering verticality
    std::size_t n_ver = bp::len(xyz);
    std::vector< std::vector< float > > geof(n_ver, std::vector< float >(4,0));
    //--- read numpy array data---
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * xyz_data = reinterpret_cast<float*>(xyz.get_data());
    std::size_t s_ver = 0;
    #pragma omp parallel for schedule(static)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++)
    {//each point can be treated in parallell independently
        //--- compute 3d covariance matrix of neighborhood ---
        ei::MatrixXf position(k_nn+1,3);
        std::size_t i_edg = k_nn * i_ver;
        std::size_t ind_nei;
        position(0,0) = xyz_data[3 * i_ver];
        position(0,1) = xyz_data[3 * i_ver + 1];
        position(0,2) = xyz_data[3 * i_ver + 2];
        for (std::size_t i_nei = 0; i_nei < k_nn; i_nei++)
        {
                //add the neighbors to the position matrix
            ind_nei = target_data[i_edg];
            position(i_nei+1,0) = xyz_data[3 * ind_nei];
            position(i_nei+1,1) = xyz_data[3 * ind_nei + 1];
            position(i_nei+1,2) = xyz_data[3 * ind_nei + 2];
            i_edg++;
        }
        // compute the covariance matrix
        ei::MatrixXf centered_position = position.rowwise() - position.colwise().mean();
        ei::Matrix3f cov = (centered_position.adjoint() * centered_position) / float(k_nn + 1);
        ei::EigenSolver<Matrix3f> es(cov);
        //--- compute the eigen values and vectors---
        std::vector<float> ev = {es.eigenvalues()[0].real(),es.eigenvalues()[1].real(),es.eigenvalues()[2].real()};
        std::vector<int> indices(3);
        std::size_t n(0);
        std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });
        std::sort(std::begin(indices),std::end(indices),
                       [&](int i1, int i2) { return ev[i1] > ev[i2]; } );
        std::vector<float> lambda = {(std::max(ev[indices[0]],0.f)),
                                    (std::max(ev[indices[1]],0.f)),
                                    (std::max(ev[indices[2]],0.f))};
        ei::Vector3f v1 = {es.eigenvectors().col(indices[0])(0).real()
                               , es.eigenvectors().col(indices[0])(1).real()
                               , es.eigenvectors().col(indices[0])(2).real()};
        ei::Vector3f v2 = {es.eigenvectors().col(indices[1])(0).real()
                               , es.eigenvectors().col(indices[1])(1).real()
                               , es.eigenvectors().col(indices[1])(2).real()};
        ei::Vector3f v3 = {es.eigenvectors().col(indices[2])(0).real()
                               , es.eigenvectors().col(indices[2])(1).real()
                               , es.eigenvectors().col(indices[2])(2).real()};

        //--- compute the dimensionality features---
        float linearity  = (sqrtf(lambda[0]) - sqrtf(lambda[1])) / sqrtf(lambda[0]);
        float planarity  = (sqrtf(lambda[1]) - sqrtf(lambda[2])) / sqrtf(lambda[0]);
        float scattering =  sqrtf(lambda[2]) / sqrtf(lambda[0]);

        std::vector<float> unary_vector =
            {lambda[0] * fabsf(v1[0]) + lambda[1] * fabsf(v2[0]) + lambda[2] * fabsf(v3[0])
            ,lambda[0] * fabsf(v1[1]) + lambda[1] * fabsf(v2[1]) + lambda[2] * fabsf(v3[1])
            ,lambda[0] * fabsf(v1[2]) + lambda[1] * fabsf(v2[2]) + lambda[2] * fabsf(v3[2])};
        float norm = sqrt(unary_vector[0] * unary_vector[0] + unary_vector[1] * unary_vector[1]
                        + unary_vector[2] * unary_vector[2]);
        float verticality = unary_vector[2] / norm;

        /* ei::Matrix3f s;

        s.row(0) = v1;
        s.row(1) = v2;
        s.row(2) = v3;

        ei::EigenSolver<Matrix3f> esS(s);
        //--- compute the eigen values and vectors---
        std::vector<float> evS = {esS.eigenvalues()[0].real(),esS.eigenvalues()[1].real(),esS.eigenvalues()[2].real()};

        std::vector<int> tindiceS(3);
        std::size_t ns(0);
        std::generate(std::begin(tindiceS), std::end(tindiceS), [&]{ return ns++; });
        std::sort(std::begin(tindiceS),std::end(tindiceS),
                       [&](int j1, int j2) { return evS[j1] > evS[j2]; } );
        std::vector<float> lambdaS = {(std::max(evS[tindiceS[0]],0.f)),                       
                                    (std::max(evS[tindiceS[1]],0.f)),                     
                                    (std::max(evS[tindiceS[2]],0.f))};
        ei::Vector3f e1 = {esS.eigenvectors().col(tindiceS[0])(0).real()                
                            , esS.eigenvectors().col(tindiceS[0])(1).real()                
                            , esS.eigenvectors().col(tindiceS[0])(2).real()};         */

        //---fill the geof vector---
        
        geof[i_ver][0] = linearity;
        geof[i_ver][1] = planarity;
        geof[i_ver][2] = scattering;
        geof[i_ver][3] = verticality;
        /* geof[i_ver][4] = float(e1[0]);
        geof[i_ver][5] = float(e1[1]);
        geof[i_ver][6] = float(e1[2]); */


        //---progression---
        s_ver++;//if run in parellel s_ver behavior is udnefined, but gives a good indication of progress

    }
    return VecvecToArray<float>::convert(geof);
}

using namespace boost::python;
BOOST_PYTHON_MODULE(libply_c)
{
    _import_array();
    bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
    Py_Initialize();
    bpn::initialize();

    def("compute_norm", compute_norm);


}