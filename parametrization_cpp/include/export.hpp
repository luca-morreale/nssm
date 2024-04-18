
#include <cstdlib>
#include <iostream>
#include <fstream>


template<typename SurfaceMesh, typename UV_pmap>
bool write_obj(std::ofstream &out, SurfaceMesh & sm, UV_pmap uv_map)  
{
    typedef typename boost::graph_traits<SurfaceMesh>::vertex_descriptor    vertex_descriptor;
    typedef typename boost::graph_traits<SurfaceMesh>::face_descriptor      face_descriptor;
    typedef typename boost::graph_traits<SurfaceMesh>::halfedge_descriptor  halfedge_descriptor;
    typedef boost::unordered_map<vertex_descriptor, std::size_t> Vertex_index_map;
    
    std::size_t vertices_counter = 0, faces_counter = 0;
    
    Vertex_index_map vium;
    boost::associative_property_map<Vertex_index_map> vimap(vium);
    
    for(auto vd : sm.vertices()){
        out << "v " << sm.point(vd) << std::endl;
    }
    
    typename SurfaceMesh::Vertex_range::iterator  vb, ve;
    for(boost::tie(vb, ve) = sm.vertices(); vb != ve; ++vb){
        auto uv = get(uv_map, *vb);
        
        out << "vt " << (uv.x()*2.0 -1.0) << " " << (uv.y()*2.0 -1.0) << std::endl;
    }
    
    typename boost::graph_traits<SurfaceMesh>::vertex_iterator vit, vend;
    boost::tie(vit, vend) = vertices(sm);
    while(vit!=vend)
    {
      vertex_descriptor vd = *vit++;
      put(vimap, vd, vertices_counter++);
    }

    BOOST_FOREACH(face_descriptor fd, faces(sm)){
      halfedge_descriptor hd = halfedge(fd, sm);
      out << "f";
      BOOST_FOREACH(vertex_descriptor vd, vertices_around_face(hd, sm)){
        out << " " << (get(vimap, vd)+1) << "/" << (get(vimap, vd)+1) << "/" << (get(vimap, vd)+1);
      }
      out << '\n';
      faces_counter++;
    }
    if(vertices_counter != sm.number_of_vertices())
      return 0;
    else if(faces_counter != sm.number_of_faces())
      return 0;
    else
      return 1;
}
