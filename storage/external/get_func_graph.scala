@main def exec(filename: String) = {
   importCode.c(filename)
   run.ossdataflow
   cpg.graph.E.map { edge =>
     List(
       edge.outNode.id,
       edge.inNode.id,
       edge.label,
       edge.propertyOption("VARIABLE").getOrElse(null)
     )
   }.toJson |> filename + ".edges.json"
   cpg.graph.V.map(node=>node).toJson |> filename + ".nodes.json"
   delete
}

