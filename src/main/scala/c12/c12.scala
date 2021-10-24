package c12


import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation._
import scalismo.mesh._
import scalismo.registration._
import scalismo.io.MeshIO
import scalismo.numerics._
import scalismo.kernels._
import scalismo.statisticalmodel._
import breeze.linalg.DenseVector

import scalismo.ui.api._

import breeze.linalg.{DenseVector}
object c12 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    /** begin */

    /** 加载和预处理数据集： */
    val referenceMesh = MeshIO.readMesh(new java.io.File("datasets/quickstart/facemesh.ply")).get
    //加载参考网格

    val modelGroup = ui.createGroup("model")
    val refMeshView = ui.show(modelGroup, referenceMesh, "referenceMesh")//显示网格
    refMeshView.color = java.awt.Color.RED

    /** Building a Gaussian process shape model */
    /** 高斯核参数选择 */
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val kernel = DiagonalKernel3D(GaussianKernel3D(sigma = 70) * 50.0, outputDim = 3)
    val gp = GaussianProcess(zeroMean, kernel)

    val interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh,
      gp,
      relativeTolerance = 0.05,
      interpolator = interpolator)//获取低秩近似

    val gpView = ui.addTransformation(modelGroup, lowRankGP, "gp")//将GP表示的变换应用于modelGroup

    /** Registration 配准*/
    /** 目标载入 */
    val targetGroup = ui.createGroup("target")
    val targetMesh = MeshIO.readMesh(new java.io.File("datasets/quickstart/face-2.ply")).get
    val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")
    /**  使用高斯过程对可能的变换进行建模*/
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)

    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = FixedPointsUniformMeshSampler3D(referenceMesh, numberOfPoints = 1000)
    val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    //评价指标，相当于cost function
    val optimizer = LBFGSOptimizer(maxNumberOfIterations = 100)

    val regularizer = L2Regularizer(transformationSpace)

    val registration = Registration(metric, regularizer, regularizationWeight = 1e-5, optimizer)

    val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)
    val registrationIterator = registration.iterator(initialCoefficients)

    val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
      println(s"object value in iteration $itnum is ${it.value}")
      gpView.coefficients = it.parameters
      it
    }

    val registrationResult = visualizingRegistrationIterator.toSeq.last

    val registrationTransformation = transformationSpace.transformationForParameters(registrationResult.parameters)
    val fittedMesh = referenceMesh.transform(registrationTransformation)

    /** Working with the registration result */
    val targetMeshOperations = targetMesh.operations
    val projection = (pt: Point[_3D]) => {
      targetMeshOperations.closestPointOnSurface(pt).point
    }

    val finalTransformation = registrationTransformation.andThen(projection)

    val projectedMesh = referenceMesh.transform(finalTransformation)
    val resultGroup = ui.createGroup("result")
    val projectionView = ui.show(resultGroup, projectedMesh, "projection")

    /** Improving registrations for more complex shapes */
    case class RegistrationParameters(regularizationWeight: Double, numberOfIterations: Int, numberOfSampledPoints: Int)

    def doRegistration(
                        lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
                        referenceMesh: TriangleMesh[_3D],
                        targetmesh: TriangleMesh[_3D],
                        registrationParameters: RegistrationParameters,
                        initialCoefficients: DenseVector[Double]
                      ): DenseVector[Double] = {
      val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
      val fixedImage = referenceMesh.operations.toDistanceImage
      val movingImage = targetMesh.operations.toDistanceImage
      val sampler = FixedPointsUniformMeshSampler3D(
        referenceMesh,
        registrationParameters.numberOfSampledPoints
      )
      val metric = MeanSquaresMetric(
        fixedImage,
        movingImage,
        transformationSpace,
        sampler
      )
      val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
      val regularizer = L2Regularizer(transformationSpace)
      val registration = Registration(
        metric,
        regularizer,
        registrationParameters.regularizationWeight,
        optimizer
      )
      val registrationIterator = registration.iterator(initialCoefficients)
      val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
        println(s"object value in iteration $itnum is ${it.value}")
        it
      }
      val registrationResult = visualizingRegistrationIterator.toSeq.last
      registrationResult.parameters
    }

    val registrationParameters = Seq(
      RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 20, numberOfSampledPoints = 1000),
      RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 30, numberOfSampledPoints = 1000),
      RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 40, numberOfSampledPoints = 2000),
      RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50, numberOfSampledPoints = 4000)
    )


    val finalCoefficients = registrationParameters.foldLeft(initialCoefficients)((modelCoefficients, regParameters) =>
      doRegistration(lowRankGP, referenceMesh, targetMesh, regParameters, modelCoefficients)
    )



    /** end */

  }
}