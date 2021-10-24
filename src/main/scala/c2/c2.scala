package c2

import java.awt.Color
import scalismo.geometry._
import scalismo.common._
import scalismo.mesh.TriangleMesh
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.ui.api._
import scalismo.transformations._
import scalismo.registration.LandmarkRegistration

object c2 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    val paolaGroup = ui.createGroup("paola")
    val mesh: TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("datasets/Paola.ply")).get
    val meshView = ui.show(paolaGroup, mesh, "Paola")
    val flipTransform = Transformation((p: Point[_3D]) => Point3D(-p.x, p.y, p.z))
    val pt: Point[_3D] = flipTransform(Point3D(1.0, 1.0, 1.0))
    val translation = Translation3D(EuclideanVector3D(100, 0, 0))
    val rotationCenter = Point3D(0.0, 0.0, 0.0)
    val rotation: Rotation[_3D] = Rotation3D(0f, 3.14f, 0f, rotationCenter)
    val pt2: Point[_3D] = rotation(Point(1, 1, 1))
    // pt2: Point[_3D] = Point3D(-0.9984061838821647, 1.0, -1.0015912799070552)
    val translatedPaola: TriangleMesh[_3D] = mesh.transform(translation)
    val paolaMeshTranslatedView = ui.show(paolaGroup, translatedPaola, "translatedPaola")
    val rigidTransform1 = CompositeTransformation(translation, rotation)
    val rigidTransform2 : RigidTransformation[_3D] = TranslationAfterRotation3D(translation, rotation)
    val paolaTransformedGroup = ui.createGroup("paolaTransformed")
    val paolaTransformed = mesh.transform(rigidTransform2)
    ui.show(paolaTransformedGroup, paolaTransformed, "paolaTransformed")
    val landmarks : Seq[Landmark[_3D]] = LandmarkIO.readLandmarksJson3D(new java.io.File("landmarks.json")).get
    val ptIds = Seq(PointId(2213), PointId(14727), PointId(8320), PointId(48182))
    val paolaLandmarks = ptIds.map(pId => Landmark(s"lm-${pId.id}", mesh.pointSet.point(pId)))
    val paolaTransformedLandmarks = ptIds.map(pId => Landmark(s"lm-${pId.id}", paolaTransformed.pointSet.point(pId)))

    val paolaLandmarkViews = paolaLandmarks.map(lm => ui.show(paolaGroup, lm, s"${lm.id}"))
    val paolaTransformedLandmarkViews = paolaTransformedLandmarks.map(lm => ui.show(paolaTransformedGroup, lm, lm.id))


    val bestTransform : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(paolaLandmarks, paolaTransformedLandmarks, center = Point(0, 0, 0))
    val transformedLms = paolaLandmarks.map(lm => lm.transform(bestTransform))
    val landmarkViews = ui.show(paolaGroup, transformedLms, "transformedLMs")
    val alignedPaola = mesh.transform(bestTransform)
    val alignedPaolaView = ui.show(paolaGroup, alignedPaola, "alignedPaola")
    alignedPaolaView.color = java.awt.Color.RED
  }
}
