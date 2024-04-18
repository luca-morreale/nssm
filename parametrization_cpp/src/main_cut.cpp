
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/Seam_mesh.h>
#include <CGAL/Surface_mesh_parameterization/IO/File_off.h>
#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/Surface_mesh_parameterization/Square_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Barycentric_mapping_parameterizer_3.h>
#include <CGAL/Unique_hash_map.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Inverse_index.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <CGAL/Surface_mesh_parameterization/ARAP_parameterizer_3.h>

#include <CGAL/IO/OFF.h>

#include <CGAL/Surface_mesh_parameterization/internal/Containers_filler.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>


typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2                Point_2;
typedef Kernel::Point_3                Point_3;
typedef CGAL::Polyhedron_3<Kernel>     PolyMesh;

typedef boost::graph_traits<PolyMesh>::edge_descriptor     SM_edge_descriptor;
typedef boost::graph_traits<PolyMesh>::halfedge_descriptor SM_halfedge_descriptor;
typedef boost::graph_traits<PolyMesh>::vertex_descriptor   SM_vertex_descriptor;

typedef CGAL::Unique_hash_map<SM_edge_descriptor, bool> Seam_edge_uhm;
typedef boost::associative_property_map<Seam_edge_uhm>  Seam_edge_pmap;

typedef CGAL::Unique_hash_map<SM_vertex_descriptor, bool> Seam_vertex_uhm;
typedef boost::associative_property_map<Seam_vertex_uhm>  Seam_vertex_pmap;

typedef CGAL::Unique_hash_map<SM_halfedge_descriptor, Point_2> UV_uhm;
typedef boost::associative_property_map<UV_uhm>                UV_pmap;

typedef CGAL::Seam_mesh<PolyMesh, Seam_edge_pmap, Seam_vertex_pmap> SeamMesh;

typedef boost::graph_traits<SeamMesh>::vertex_descriptor   vertex_descriptor;
typedef boost::graph_traits<SeamMesh>::halfedge_descriptor halfedge_descriptor;
typedef boost::graph_traits<SeamMesh>::face_descriptor     face_descriptor;

typedef boost::unordered_map<vertex_descriptor, std::size_t> Vertex_index_map;

namespace SMP = CGAL::Surface_mesh_parameterization;

typedef SMP::Square_border_uniform_parameterizer_3<SeamMesh>  Border_parameterizer;
typedef SMP::Barycentric_mapping_parameterizer_3<SeamMesh, Border_parameterizer> Parameterizer;


struct Compute_area:
  public std::unary_function<const PolyMesh::Facet, double>
{
  double operator()(const PolyMesh::Facet& f) const{
    return Kernel::Compute_area_3()(
      f.halfedge()->vertex()->point(),
      f.halfedge()->next()->vertex()->point(),
      f.halfedge()->opposite()->vertex()->point() );
  }
};

void write_obj(std::ofstream &out, SeamMesh &mesh, UV_pmap &uv_pm);
void check_facets_area(SeamMesh &mesh, UV_pmap &uv_pm, halfedge_descriptor &bhd);

int main(int argc, char** argv)
{
        if (argc < 3) {
        std::cerr << "ERROR! Files name missing." << std::endl;
        return 1;
    }
    
    std::string file(argv[1]);
    std::ifstream in(argv[1]);
    if(!in) {
        std::cerr << "Problem loading the input data" << std::endl;
        return EXIT_FAILURE;
    }
    
    // read mesh
    PolyMesh sm;
    in >> sm;
    
    if (sm.empty()) {
        std::cerr << "Empty Polyhedron, your model might not be manifold." << std::endl;
        return 1;
    }
        
    // create seam mesh object
    Seam_edge_uhm seam_edge_uhm(false);
    Seam_edge_pmap seam_edge_pm(seam_edge_uhm);
    
    Seam_vertex_uhm seam_vertex_uhm(false);
    Seam_vertex_pmap seam_vertex_pm(seam_vertex_uhm);
    
    SeamMesh mesh(sm, seam_edge_pm, seam_vertex_pm);
    
    // read seam from file
    const char* filename = argv[2];
    SM_halfedge_descriptor smhd = mesh.add_seams(filename);
    if(smhd == SM_halfedge_descriptor() ) {
        std::cerr << "Warning: No seams in input" << std::endl;
    }
    
    // A halfedge on the (possibly virtual) border
    halfedge_descriptor bhd = CGAL::Polygon_mesh_processing::longest_border(mesh, CGAL::Polygon_mesh_processing::parameters::all_default()).first;
    
    // create parametrization map
    UV_uhm uv_uhm;
    UV_pmap uv_pm(uv_uhm);
        
    // parametrization
    Parameterizer param = Parameterizer();
    SMP::parameterize(mesh, param, bhd, uv_pm);

    // save mesh parametrized
    std::string out_file = file.substr(0, file.size()-4) + "_cut.obj";
    std::ofstream out(out_file);
    write_obj(out, mesh, uv_pm);
    
    check_facets_area(mesh, uv_pm, bhd);
    
    return EXIT_SUCCESS;

}

void write_obj(std::ofstream &out, SeamMesh &mesh, UV_pmap &uv_pm)
{
    Vertex_index_map vium;
    boost::associative_property_map<Vertex_index_map> vimap(vium);
    std::size_t vertices_counter = 0, faces_counter = 0;
    
    // save vertices and texture coordinates
    boost::property_map<SeamMesh, CGAL::vertex_point_t>::type vpm = get(CGAL::vertex_point, mesh);
    boost::graph_traits<SeamMesh>::vertex_iterator vb, ve;
    for(boost::tie(vb, ve) = vertices(mesh); vb != ve; ++vb)
    {
        vertex_descriptor vd = *vb;
        halfedge_descriptor hd = halfedge(vd, mesh);
        
        auto pt = get(vpm, target(hd, mesh));
        //auto pt = get(vpm, source(hd, mesh));
        auto uv = get(uv_pm, hd); 
        out << "v "  << pt << std::endl;
        out << "vt " << -(uv.x()* 2.0 - 1.0) << " " << (uv.y()* 2.0 - 1.0) << std::endl;

        // set index to vertices
        put(vimap, vd, vertices_counter++);
    }

    // faces
    BOOST_FOREACH(face_descriptor fd, faces(mesh)) {
        halfedge_descriptor hd = halfedge(fd, mesh);
        out << "f";
        BOOST_FOREACH(vertex_descriptor vd, vertices_around_face(hd, mesh)){
            auto idx = get(vimap, vd) + 1;
            out << " " << idx << "/" << idx << "/" << idx;
        }
        out << std::endl;
    }
}


void check_facets_area(SeamMesh &mesh, UV_pmap &uv_pm, halfedge_descriptor &bhd)
{
    std::stringstream out;
    SMP::IO::output_uvmap_to_off(mesh, bhd, uv_pm, out);
    
    PolyMesh tmp;
    out >> tmp;
    
    Compute_area ca;
    std::size_t num_null_faces = 0;

    for (auto it = tmp.facets_begin(); it != tmp.facets_end(); it++) {
        if (ca(*it) == 0.0) {
            num_null_faces++;
        }
    }
    
    if (num_null_faces > 0) {
        std::cerr << "WARNING: " << num_null_faces << " faces have 0 area!" << std::endl;
    }

}
