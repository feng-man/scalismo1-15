package c9

import java.awt.Color

import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh._
import scalismo.io.{StatisticalModelIO, MeshIO, LandmarkIO}
import scalismo.statisticalmodel._
import scalismo.numerics.UniformMeshSampler3D
import scalismo.kernels._
import breeze.linalg.{DenseMatrix, DenseVector}

import scalismo.ui.api._
object c9 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    /** begin */
    val noseless = MeshIO.readMesh(new java.io.File("datasets/noseless.ply")).get

    val targetGroup = ui.createGroup("target")
    ui.show(targetGroup, noseless,"noseless")

    val smallModel = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/model.h5")).get
    /** Enlarging the flexibility of a shape model */

    val scalarValuedKernel = GaussianKernel3D(70) * 10.0

    case class XmirroredKernel(kernel: PDKernel[_3D]) extends PDKernel[_3D] {
      override def domain = EuclideanSpace3D
      override def k(x: Point[_3D], y: Point[_3D]) = kernel(Point(x(0) * -1f, x(1), x(2)), y)
    }

    def symmetrizeKernel(kernel: PDKernel[_3D]): MatrixValuedPDKernel[_3D] = {
      val xmirrored = XmirroredKernel(kernel)
      val k1 = DiagonalKernel(kernel, 3)
      val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)
      k1 + k2
    }

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](symmetrizeKernel(scalarValuedKernel))

    val lowrankGP = LowRankGaussianProcess.approximateGPCholesky(
      smallModel.reference,
      gp,
      relativeTolerance = 0.5e-1,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )//相当于用近似采样获得样本
    val model = PointDistributionModel.augmentModel(smallModel, lowrankGP)//生成点分布模型并加入到参考面small

    val modelGroup = ui.createGroup("face model")
    val ssmView = ui.show(modelGroup, model, "model")

    val referenceLandmarks = LandmarkIO.readLandmarksJson3D(new java.io.File("datasets/modelLandmarks.json")).get
    val referencePoints : Seq[Point[_3D]] = referenceLandmarks.map(lm => lm.point)
    val referenceLandmarkViews = referenceLandmarks.map(lm => ui.show(modelGroup, lm, s"lm-${lm.id}"))


    val noselessLandmarks = LandmarkIO.readLandmarksJson3D(new java.io.File("datasets/noselessLandmarks.json")).get
    val noselessPoints : Seq[Point[_3D]] = noselessLandmarks.map(lm => lm.point)
    val noselessLandmarkViews = noselessLandmarks.map(lm => ui.show(targetGroup, lm, s"lm-${lm.id}"))

    val domain = UnstructuredPointsDomain3D(referencePoints.toIndexedSeq)
    val deformations = (0 until referencePoints.size).map(i => noselessPoints(i) - referencePoints(i) )
    val defField = DiscreteField3D(domain, deformations)
    ui.show(modelGroup, defField, "partial_Field")

    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.5)

    val regressionData = for ((refPoint, noselessPoint) <- referencePoints zip noselessPoints) yield {
      val refPointId = model.reference.pointSet.findClosestPoint(refPoint).id
      (refPointId, noselessPoint, littleNoise)
    }//zip是将前后两个对象组合起来，<-是赋值操作，yield{}是返回for循环中的值。这个值为{}里面的内容

    val posterior = model.posterior(regressionData.toIndexedSeq)

    val posteriorGroup = ui.createGroup("posterior-model")
    ui.show(posteriorGroup, posterior, "posterior")


    /** end  */

    /** end  */
  }
}
