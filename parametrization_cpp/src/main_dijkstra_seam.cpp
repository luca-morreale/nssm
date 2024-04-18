
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <boost/variant.hpp>
#include <boost/lexical_cast.hpp>
#include <CGAL/boost/graph/dijkstra_shortest_paths.h>

typedef CGAL::Simple_cartesian<double>      Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> Triangle_mesh;
typedef boost::graph_traits<Triangle_mesh>  Graph_traits;
typedef Graph_traits::vertex_descriptor     vertex_descriptor;

typedef std::vector<vertex_descriptor>                  VertexDescriptorList;
typedef std::map<vertex_descriptor, int>                VertexIndexMap;
typedef boost::associative_property_map<VertexIndexMap> VertexIdPropertyMap;

typedef boost::iterator_property_map<VertexDescriptorList::iterator, VertexIdPropertyMap> PredecessorMap;
typedef boost::iterator_property_map<std::vector<double>::iterator, VertexIdPropertyMap>  DistanceMap;


int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "ERROR: need to specify .off and selection files" << std::endl;
        return 1;
    }
    
    // read mesh
    Triangle_mesh tmesh;
    std::ifstream input(argv[1]);
    input >> tmesh;
    input.close();
    
    VertexIndexMap vertex_id_map;
    VertexIdPropertyMap vertex_index_pmap(vertex_id_map);
    int index = 0;
    for(vertex_descriptor vd : vertices(tmesh)) {
        vertex_id_map[vd] = index++;
    }
    
    input.open(argv[2]);

    std::ofstream out(std::string(argv[1])+".selection.txt");
    out << std::endl << std::endl;
    
    while (!input.eof()) {
        
        int start, end;
        input >> start >> end;
        
        vertex_descriptor vstart(start);
        vertex_descriptor vend(end);
        
        // We first declare a vector
        std::vector<vertex_descriptor> predecessor(num_vertices(tmesh));
        // and then turn it into a property map
        std::vector<double> distance(num_vertices(tmesh));
        PredecessorMap predecessor_pmap(predecessor.begin(), vertex_index_pmap);
        DistanceMap distance_pmap(distance.begin(), vertex_index_pmap);
        
        boost::dijkstra_shortest_paths(tmesh, vstart,
                                 distance_map(distance_pmap)
                                 .predecessor_map(predecessor_pmap)
                                 .vertex_index_map(vertex_index_pmap));
        
        vertex_descriptor it = vend;
        
        out << vertex_id_map[it] << " ";
        it = boost::get(predecessor_pmap, it);
        
        while (it != vstart) {
            out << vertex_id_map[it] << " " << vertex_id_map[it] << " ";
            it = boost::get(predecessor_pmap, it);
        }
        out << vertex_id_map[it] << " ";
        
    }
    input.close();
    

    return 0;
}
