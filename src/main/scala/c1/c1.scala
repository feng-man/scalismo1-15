package c1

import java.awt.Color
import scalismo.ui.api.ScalismoUI
import scalismo.mesh.TriangleMesh
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.common.PointId
import scalismo.mesh.TriangleId
import scalismo.geometry.{Point3D, _3D}
import scalismo.image.{DiscreteImage, DiscreteImage3D}
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.geometry.Point
import scalismo.geometry.EuclideanVector
import scalismo.io.ImageIO
import scalismo.geometry.{IntVector, IntVector3D}
import scalismo.io.StatisticalModelIO
import scalismo.ui.api.LandmarkView
import scalismo.geometry.Landmark
object c1 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    // read a mesh from file
    //val mesh = MeshIO.readMesh(new File("data/facemesh.ply")).get
    val mesh: TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("datasets/Paola.ply")).get
    // display it
    //val meshView = ui.show(mesh, "face")
    val paolaGroup = ui.createGroup("paola")
    val meshView = ui.show(paolaGroup, mesh, "Paola")
    // change its color
    meshView.color = Color.PINK

    //val pointCloudView = ui.show(paolaGroup, mesh.pointSet, "pointCloud")
    val p1: Point[_3D] = Point3D(4.0, 5.0, 6.0)
    val p2: Point[_3D] = Point3D(1.0, 2.0, 3.0)

    val v1: EuclideanVector[_3D] = Point3D(4.0, 5.0, 6.0) - Point3D(1.0, 2.0, 3.0)
    val p3: Point[_3D] = p1 + v1
    val v2: EuclideanVector[_3D] = p1.toVector
    val v3: Point[_3D] = v1.toPoint
    val pointList = Seq(
      Point3D(4.0, 5.0, 6.0),
      Point3D(1.0, 2.0, 3.0),
      Point3D(14.0, 15.0, 16.0),
      Point3D(7.0, 8.0, 9.0),
      Point3D(10.0, 11.0, 12.0)
    )
    val vectors = pointList.map { p: Point[_3D] => p.toVector } // use map to turn points into vectors
    val vectorSum = vectors.reduce { (v1, v2) => v1 + v2 } // sum up all vectors in the collection
    val centerV: EuclideanVector[_3D] = vectorSum * (1.0 / pointList.length) // divide the sum by the number of points
    val image: DiscreteImage[_3D, Short] = ImageIO.read3DScalarImage[Short](new java.io.File("datasets/PaolaMRI.vtk")).get
    val imageView = ui.show(paolaGroup, image, "mri")

    val origin: Point[_3D] = image.domain.origin
    val spacing: EuclideanVector[_3D] = image.domain.spacing
    val size: IntVector[_3D] = image.domain.size
    val imagePoints: Iterator[Point[_3D]] = image.domain.pointSet.points.take(10)
    val gridPointsView = ui.show(paolaGroup, imagePoints.toIndexedSeq, "imagePoints")
    println(origin)
    val values : Iterator[Short] = image.values
    image.values.next
    // res4: Short = 0
    image(IntVector(0,0,0))
    // res5: Short = 0
    image.values.size == image.domain.pointSet.numberOfPoints

    val threshValues = image.values.map { v: Short => if (v <= 300) v else 0.toShort }
    val thresholdedImage: DiscreteImage[_3D, Short] = DiscreteImage3D[Short](image.domain, threshValues.toIndexedSeq)
    ui show(paolaGroup, thresholdedImage, "thresh")



    val faceModel: PointDistributionModel[_3D, TriangleMesh] = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/bfm.h5")).get
    val faceModelView = ui.show(faceModel, "faceModel")

    val randomFace: TriangleMesh[_3D] = faceModel.sample
    val randfaceview = ui.show(randomFace, name = "mycode")

    val matchingLandmarkViews : Seq[LandmarkView] = ui.filter[LandmarkView](paolaGroup, (l : LandmarkView) => l.name == "noseLM")
    val matchingLandmarks : Seq[Landmark[_3D]] = matchingLandmarkViews.map(lmView => lmView.landmark)

    val landmarkId : String = matchingLandmarks.head.id
    val landmarkPosition : Point[_3D] = matchingLandmarks.head.point
  }
}
