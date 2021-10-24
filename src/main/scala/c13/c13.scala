package c13


import java.awt.Color

import scalismo.geometry._
import scalismo.transformations._
import scalismo.registration._
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.asm._
import scalismo.io.{ActiveShapeModelIO, ImageIO}

import scalismo.ui.api._
import breeze.linalg.{DenseVector}
import breeze.linalg.{DenseMatrix, DenseVector}
object c13 {

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

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("datasets/femur-asm.h5")).get

    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

    val profiles = asm.profiles
    profiles.map(profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })

    val image = ImageIO.read3DScalarImage[Short](new java.io.File("datasets/femur-image.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")

    val imageView = ui.show(targetGroup, image, "image")

    val preprocessedImage = asm.preprocessor(image)

    val point1 = image.domain.origin + EuclideanVector3D(10.0, 10.0, 10.0)
    val profile = asm.profiles.head
    val feature1 : DenseVector[Double] = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get
    //4个参数，ASM预处理的图像、图像中要计算特征向量的点、ASM中统计模型的平均形状、ASM强度模型轮廓中点的ID
    val point2 = image.domain.origin + EuclideanVector3D(20.0, 10.0, 10.0)
    val featureVec1 = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get
    val featureVec2 = asm.featureExtractor(preprocessedImage, point2, asm.statisticalModel.mean, profile.pointId).get

    val probabilityPoint1 = profile.distribution.logpdf(featureVec1)
    val probabilityPoint2 = profile.distribution.logpdf(featureVec2)
    //检索每个对应点对应于给定轮廓点的可能性，由此决定哪个点更可能对应ASM模型中的点，并以此为目标函数训练模型
    println(probabilityPoint1)
    println(probabilityPoint2)
    /** The original Active Shape Model Fitting */
    /** 为了配置拟合过程，我们需要设置一个搜索方法，它搜索给定的模型点，
     * 图像中的对应点。从这些点中，选择最有可能的点作为算法一次迭代的对应点。*/
    /** Scalismo 中已经实现的一种搜索策略是沿着模型点的法线方向进行搜索。
     * 此行为由NormalDirectionSearchPointSampler */
    val searchSampler = NormalDirectionSearchPointSampler(numberOfPoints = 100, searchDistance = 3)
    /** 设置拟合过程中的参数 */
    val config = FittingConfiguration(featureDistanceThreshold = 3, pointDistanceThreshold = 5, modelCoefficientBounds = 3)
    // make sure we rotate around a reasonable center point
    val modelBoundingBox = asm.statisticalModel.referenceMesh.boundingBox
    val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

    // we start with the identity transform
    val translationTransformation = Translation3D(EuclideanVector3D(0, 0, 0))
    val rotationTransformation = Rotation3D(0, 0, 0, rotationCenter)
    val initialRigidTransformation = TranslationAfterRotation3D(translationTransformation, rotationTransformation)
    val initialModelCoefficients = DenseVector.zeros[Double](asm.statisticalModel.rank)
    val initialTransformation = ModelTransformations(initialModelCoefficients, initialRigidTransformation)

    val numberOfIterations = 20
    val asmIterator = asm.fitIterator(image, searchSampler, numberOfIterations, config, initialTransformation)

    val asmIteratorWithVisualization = asmIterator.map(it => {
      it match {
        case scala.util.Success(iterationResult) => {
          modelView.shapeModelTransformationView.poseTransformationView.transformation = iterationResult.transformations.rigidTransform
          modelView.shapeModelTransformationView.shapeTransformationView.coefficients = iterationResult.transformations.coefficients
        }
        case scala.util.Failure(error) => System.out.println(error.getMessage)
      }
      it
    })
    val result = asmIteratorWithVisualization.toIndexedSeq.last
    val finalMesh = result.get.mesh


    /** Evaluating the likelihood of a model instance under the image */
    def likelihoodForMesh(asm : ActiveShapeModel, mesh : TriangleMesh[_3D], preprocessedImage: PreprocessedImage) : Double = {

      val ids = asm.profiles.ids

      val likelihoods = for (id <- ids) yield {
        val profile = asm.profiles(id)
        val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
        val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).get
        profile.distribution.logpdf(featureAtPoint)
      }
      likelihoods.sum
    }
    //分别对ASM中的统计模型进行采样，得到两网格
    val sampleMesh1 = asm.statisticalModel.sample
    val sampleMesh2 = asm.statisticalModel.sample
    //比较mesh1、mesh2和目标图像对应的可能性
    println("Likelihood for mesh 1 = " + likelihoodForMesh(asm, sampleMesh1, preprocessedImage))
    println("Likelihood for mesh 2 = " + likelihoodForMesh(asm, sampleMesh2, preprocessedImage))


    /** end */

    /** end  */
  }
}
