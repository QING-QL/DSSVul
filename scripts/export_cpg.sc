import io.shiftleft.semanticcpg.language._
import scala.jdk.CollectionConverters._
import java.nio.file.{Files, Paths}

@main def exportGraphs(cpgFile: String, outFile: String) = {
  importCpg(cpgFile)
  if (!cpg.metaData.overlays.l.flatten.exists(x => x == "ossdataflow" || x == "dataflowOss")) {
    run.ossdataflow
  }
  val wanted = Set("AST", "CFG", "REACHING_DEF", "CDG")
  val methods = cpg.method.filterNot(_.isExternal).l.groupBy(_.filename)
  val rows = methods.toSeq.sortBy(_._1).flatMap { case (filename, ms) =>
    val nodes = ms.flatMap(m => m.ast.l).groupBy(_.id).values.map(_.head).toSeq.sortBy(_.id)
    val ids = nodes.map(_.id).toSet
    if (nodes.isEmpty || filename == "<empty>") None
    else {
      val edges = nodes.flatMap { n =>
        wanted.toSeq.sorted.flatMap { relation =>
          n.outE(relation).filter(e => ids.contains(e.dst.id)).map { e =>
            ujson.Arr(n.id.toString, e.dst.id.toString, relation)
          }.toSeq
        }
      }
      Some(ujson.Obj("file" -> filename,
        "nodes" -> ujson.Arr.from(nodes.map(n => ujson.Obj("id" -> n.id.toString, "type" -> n.label, "code" -> n.code))),
        "edges" -> ujson.Arr.from(edges)))
    }
  }
  Files.writeString(Paths.get(outFile), ujson.write(ujson.Arr.from(rows)))
  println("EXPORTED_FILES=" + rows.size)
}
