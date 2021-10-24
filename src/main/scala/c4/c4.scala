package c4

import scalismo.geometry._
import scalismo.common._
import scalismo.mesh._
import scalismo.io.{StatismoIO, StatisticalModelIO}
import scalismo.statisticalmodel._
import scalismo.ui.api._
import scalismo.ui.api.ScalismoUI
import breeze.linalg.{DenseMatrix, DenseVector}
object c4 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    /** begin */
    val faceModel : PointDistributionModel[_3D, TriangleMesh] = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("IdeaProjects/git/datasets/bfm.h5")).get
    val modelGroup = ui.createGroup("model")
    //可视化平均形状
    val sampleGroup = ui.createGroup("samples")

    val meanFace : TriangleMesh[_3D] = faceModel.mean
    ui.show(sampleGroup, meanFace, "meanFace")
    //采样获得concrete face网格
    val sampledFace : TriangleMesh[_3D] = faceModel.sample
    ui.show(sampleGroup, sampledFace, "randomFace")
    //pdm背后的高斯过程
    val reference : TriangleMesh[_3D] = faceModel.reference
    val faceGP : DiscreteLowRankGaussianProcess[_3D, TriangleMesh, EuclideanVector[_3D]] = faceModel.gp
    /**  它是一个 DiscreteGaussianProcess。这意味着，过程模型定义在离散的、有限的点集上的函数。
          它在 3D 空间中定义（由类型参数表示_3D）
          它的建模函数域是 TriangleMesh
          建模函数的值是向量（更准确地说，它们的类型是EuclideanVector）。
          它使用低秩近似表示。这是一个技术性的问题，我们稍后会回来讨论。
         因此，当我们从高斯过程中抽取样本或获得平均值时，我们期望获得具有匹
         配特征的函数。确实如此*/
    val meanDeformation : DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] = faceGP.mean
    val sampleDeformation : DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] = faceGP.sample
    //可视化平均变形
    ui.show(sampleGroup, meanDeformation, "meanField")
    /** 正如您所希望的那样，平均变形向量的所有尖端都在平均面的点上结束。
     为了找出它们从哪里开始，让我们显示面部模型的参考网格：*/
    //ui.show(modelGroup, referenceFace, "referenceFace")
    /** end  */
  }
}
