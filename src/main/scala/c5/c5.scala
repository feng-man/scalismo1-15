package c5

import java.awt.Color

import scalismo.ui.api._
import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh._
import scalismo.io.StatisticalModelIO
import scalismo.statisticalmodel._

object c5 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    /** begin */
    /** begin */
    /** 导入数据集，并在ui上显示 */
    val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/bfm.h5")).get
    val gp = model.gp

    val modelGroup = ui.createGroup("modelGroup")
    val ssmView = ui.show(modelGroup, model, "model")

    /** 对高斯过程进行采样 */
    val sampleDF : DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] = model.gp.sample

    val sampleGroup = ui.createGroup("sample")
    ui.show(sampleGroup, sampleDF, "discreteSample")

    /** 对离散的采样进行插值 */
    val interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    val contGP = model.gp.interpolate(interpolator)

    /**  */
    val contSample: Field[_3D, EuclideanVector[_3D]] = contGP.sample

    /** 从连续到离散：边缘化和离散化 */
    /** 对参考网格上所有的点进行采样 */
    val fullSample = contGP.sampleAtPoints(model.reference)
    val fullSampleView = ui.show(sampleGroup, fullSample, "fullSample")

    /**  通过marginal获取样本点的分布*/
    val referencePointSet = model.reference.pointSet
    val rightEyePt: Point[_3D] = referencePointSet.point(PointId(4281))
    val leftEyePt: Point[_3D] = referencePointSet.point(PointId(11937))
    val marginal : DiscreteGaussianProcess[_3D, UnstructuredPointsDomain, EuclideanVector[_3D]] = contGP.marginal(IndexedSeq(rightEyePt,leftEyePt))

    /** 我们可以使用该discretize方法，该方法将一个域作为参数，并产生在该域上定义的离散高斯过程。 */
    val discreteGP : DiscreteGaussianProcess[_3D, TriangleMesh, EuclideanVector[_3D]] = contGP.discretize(model.reference)

    /** 改变点分布模型的参考 */
    val lowresMesh = model.reference.operations.decimate(1000)
    val lowResModel = model.newReference(lowresMesh, TriangleMeshInterpolator3D())

    /** 形状和变形的概率： */
    val defSample = model.gp.sample
    model.gp.pdf(defSample)

    /**  */
    val defSample1 = model.gp.sample
    // defSample1: DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] = <function1>
    val defSample2 = model.gp.sample
    // defSample2: DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] = <function1>

    val logPDF1 = model.gp.logpdf(defSample1)
    // logPDF1: Double = -12.599680420141278
    val logPDF2 = model.gp.logpdf(defSample2)
    // logPDF2: Double = -12.300955026410005

    val moreOrLess = if (logPDF1 > logPDF2) "more" else "less"
    // moreOrLess: String = "less"
    println(s"defSample1 is $moreOrLess likely than defSample2")
    // defSample1 is less likely than defSample2

    /** end  */
  }
}
