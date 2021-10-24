package c11

import scalismo.geometry._
import scalismo.common._
import scalismo.mesh._
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.numerics.UniformMeshSampler3D
import scalismo.io.{MeshIO, StatisticalModelIO, LandmarkIO}

import scalismo.ui.api._

import breeze.linalg.{DenseMatrix, DenseVector}
object c11 {

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
    /** 加载和预处理数据集： */
    val targetMesh = MeshIO.readMesh(new java.io.File("datasets/target.ply")).get
    //目标网格
    val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/bfm.h5")).get
    //统计形状模型 将model拟合的目标网格
    /** 显示网格和模型 */
    val targetGroup = ui.createGroup("targetGroup")
    val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")

    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, model, "model")

    /** Iterative Closest Points (ICP) and GP regression */
    val sampler = UniformMeshSampler3D(model.reference, numberOfPoints = 5000)//获取形状模型表面5000个均匀分布的点
    val points : Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1) // we only want the points

    val ptIds = points.map(point => model.reference.pointSet.findClosestPoint(point).id)//获取采样点的点ID

    def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(PointId, Point[_3D])] = {
      ptIds.map{ id : PointId =>
        val pt = movingMesh.pointSet.point(id)
        val closestPointOnMesh2 = targetMesh.pointSet.findClosestPoint(pt).point
        (id, closestPointOnMesh2)
      }
    }//它为每个兴趣点找到目标上最近的点即：找到目标网格上最近的点


    val correspondences = attributeCorrespondences(model.mean, ptIds)//调用上面的函数获取目标网格上最近的点

    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

    def fitModel(correspondences: Seq[(PointId, Point[_3D])]) : TriangleMesh[_3D] = {
      val regressionData = correspondences.map(correspondence =>
        (correspondence._1, correspondence._2, littleNoise)
      )
      val posterior = model.posterior(regressionData.toIndexedSeq)
      posterior.mean
    }//拟合模型函数

    val fit = fitModel(correspondences)
    val resultGroup = ui.createGroup("results")
    val fitResultView = ui.show(resultGroup, fit, "fit")//显示拟合结果

    def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
      if (numberOfIterations == 0) movingMesh
      else {
        val correspondences = attributeCorrespondences(movingMesh, ptIds)
        val transformed = fitModel(correspondences)

        nonrigidICP(transformed, ptIds, numberOfIterations - 1)
      }
    }
    val finalFit = nonrigidICP( model.mean, ptIds, 20)//迭代

    ui.show(resultGroup, finalFit, "final fit")//最终结果
    /** end */

  }
}