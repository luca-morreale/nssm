#include <iostream>

#include <igl/slim.h>

#include <igl/vertex_components.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/Timer.h>

#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/MappingEnergyType.h>
#include <igl/serialize.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/flipped_triangles.h>
#include <igl/euler_characteristic.h>
#include <igl/barycenter.h>
#include <igl/adjacency_matrix.h>
#include <igl/is_edge_manifold.h>
#include <igl/doublearea.h>
#include <igl/cat.h>
#include <igl/PI.h>

#include <stdlib.h>

#include <string>
#include <vector>

using namespace std;
using namespace Eigen;

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
void writeOBJ(const std::string str,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& TC,
    const Eigen::MatrixXi& FTC);


Eigen::MatrixXd V;
Eigen::MatrixXi F;
bool first_iter = true;
igl::SLIMData sData;
igl::Timer timer;

double uv_scale_param;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Missing obj file" << std::endl;
        return 1;
    }
    std::string file(argv[1]);


    Eigen::MatrixXd V, TC, N;
    Eigen::MatrixXi F,FTC,FN;
    igl::readOBJ(file, V, TC, N, F, FTC, FN);

    Eigen::MatrixXd uv_init;
    Eigen::VectorXi bnd; Eigen::MatrixXd bnd_uv;
    Eigen::VectorXi b; Eigen::MatrixXd bc;
    igl::boundary_loop(F,bnd);
    igl::map_vertices_to_circle(V,bnd,bnd_uv);

    std::cout << "init boundary" << std::endl;


    igl::harmonic(V,F,bnd,bnd_uv,1,uv_init);
    if (igl::flipped_triangles(uv_init,F).size() != 0) {
        igl::harmonic(F,bnd,bnd_uv,1,uv_init); // use uniform laplacian
    }

    cout << "initialized parametrization" << endl;
    double soft_const_p = 1e35;
    sData.slim_energy = igl::MappingEnergyType::SYMMETRIC_DIRICHLET;
    slim_precompute(V, F, uv_init, sData, igl::MappingEnergyType::SYMMETRIC_DIRICHLET, bnd, bnd_uv, soft_const_p);
    cout << "energy = " << sData.energy << endl;
    slim_solve(sData, 100); // 10 iter

    cout << "energy = " << sData.energy << endl;

    Eigen::MatrixXd empty(0, 3);
    const std::string out_file = file.substr(0, file.size()-4) + "_slim.obj";
    std::cout << out_file << std::endl;
    writeOBJ(out_file, V, F, sData.V_o, F);

    return 0;
}

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {

    Eigen::SparseMatrix<double> A;
    igl::adjacency_matrix(F,A);

    Eigen::MatrixXi C, Ci;
    igl::vertex_components(A, C, Ci);

    int connected_components = Ci.rows();
    if (connected_components!=1) {
        cout << "Error! Input has multiple connected components" << endl; exit(1);
    }
    int euler_char = igl::euler_characteristic(V, F);
    if (euler_char!=1)
    {
        cout <<
            "Error! Input does not have a disk topology, it's euler char is " <<
            euler_char << endl;
        exit(1);
    }
    bool is_edge_manifold = igl::is_edge_manifold(F);
    if (!is_edge_manifold) {
        cout << "Error! Input is not an edge manifold" << endl; exit(1);
    }

    Eigen::VectorXd areas; igl::doublearea(V,F,areas);
    const double eps = 1e-14;
    for (int i = 0; i < areas.rows(); i++) {
        if (areas(i) < eps) {
            cout << "Error! Input has zero area faces" << endl; exit(1);
        }
    }
}


void writeOBJ(const std::string str,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& TC,
    const Eigen::MatrixXi& FTC)
{
    FILE * obj_file = fopen(str.c_str(),"w");
    if(NULL==obj_file)
    {
        printf("IOError: %s could not be opened for writing...",str.c_str());
        return;
    }
    // Loop over V
    for(int i = 0;i<(int)V.rows();i++)
    {
        fprintf(obj_file,"v");
        for(int j = 0;j<(int)V.cols();++j)
        {
            fprintf(obj_file," %0.17g", V(i,j));
        }
        fprintf(obj_file,"\n");
    }
    bool write_N = false;


    bool write_texture_coords = TC.rows() >0;

    if(write_texture_coords)
    {
        for(int i = 0;i<(int)TC.rows();i++)
        {
            fprintf(obj_file, "vt %0.17g %0.17g\n",TC(i,0),TC(i,1));
        }
        fprintf(obj_file,"\n");
    }

    // loop over F
    for(int i = 0;i<(int)F.rows();++i)
    {
        fprintf(obj_file,"f");
        for(int j = 0; j<(int)F.cols();++j)
        {
          // OBJ is 1-indexed
          fprintf(obj_file," %u",F(i,j)+1);

          if(write_texture_coords)
            fprintf(obj_file,"/%u",FTC(i,j)+1);

        }
        fprintf(obj_file,"\n");
    }
    fclose(obj_file);

}


