package c15

import java.awt.Color

import scalismo.common.{PointId, UnstructuredPointsDomain}
import scalismo.geometry._
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, PointDistributionModel, PointDistributionModel3D}
import scalismo.transformations.{
  RigidTransformation,
  Rotation3D,
  Translation3D,
  TranslationAfterRotation,
  TranslationAfterRotation3D
}

import scalismo.utils.Memoize

import scalismo.ui.api.ScalismoUI
import breeze.linalg.{DenseMatrix, DenseVector}
object c15 {

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
    /** 导入数据并显示 */
    /** 这里的model相当于初始位置 */
    val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/bfm.h5")).get

    val modelGroup = ui.createGroup("model")
    val modelView = ui.show(modelGroup, model, "model")
    modelView.referenceView.opacity = 0.5
    /** 加载目标面和实际模型对应的地标 */
    val modelLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/modelLM_mcmc.json")).get
    val modelLmViews = ui.show(modelGroup, modelLms, "modelLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.BLUE)

    val targetGroup = ui.createGroup("target")

    val targetLms = LandmarkIO.readLandmarksJson3D(new java.io.File("datasets/targetLM_mcmc.json")).get
    val targetLmViews = ui.show(targetGroup, targetLms, "targetLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)
    /** 使用点ID来引用模型上的点，使用物理点来表示目标位置 */
    //这是因为模型上的会随着拟合的变化而变化
    val modelLmIds = modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
    val targetPoints = targetLms.map(l => l.point)
    /** 将点的对应关系记录在correspondences 中 */
    val correspondences = modelLmIds
      .zip(targetPoints)
      .map(modelIdWithTargetPoint => {
        val (modelId, targetPoint) = modelIdWithTargetPoint
        (modelId, targetPoint)
      })
    /** 定义后验模型中的参数类包括平移、旋转变形参数以及噪声参数 */
    case class Parameters(translationParameters: EuclideanVector[_3D],
                          rotationParameters: (Double, Double, Double),
                          modelCoefficients: DenseVector[Double],
                          noiseStddev : Double
                         )

    case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point[_3D]) {
      def poseTransformation: TranslationAfterRotation[_3D] = {
        val translation = Translation3D(parameters.translationParameters)
        val rotation = Rotation3D(
          parameters.rotationParameters._1,
          parameters.rotationParameters._2,
          parameters.rotationParameters._3,
          rotationCenter
        )
        TranslationAfterRotation3D(translation, rotation)
      }
    }
    /** 定义先验分布 */
    case class PriorEvaluator(model: PointDistributionModel[_3D, TriangleMesh]) extends DistributionEvaluator[Sample] {

      val translationPrior = breeze.stats.distributions.Gaussian(0.0, 5.0)
      val rotationPrior = breeze.stats.distributions.Gaussian(0, 0.1)
      val noisePrior = breeze.stats.distributions.LogNormal(0, 0.25)
      /** 将先验分布表示为对数形式 */
      override def logValue(sample: Sample): Double = {
        model.gp.logpdf(sample.parameters.modelCoefficients) +
          translationPrior.logPdf(sample.parameters.translationParameters.x) +
          translationPrior.logPdf(sample.parameters.translationParameters.y) +
          translationPrior.logPdf(sample.parameters.translationParameters.z) +
          rotationPrior.logPdf(sample.parameters.rotationParameters._1) +
          rotationPrior.logPdf(sample.parameters.rotationParameters._2) +
          rotationPrior.logPdf(sample.parameters.rotationParameters._3) +
          noisePrior.logPdf(sample.parameters.noiseStddev)
      }
    }
    /** 用模型和对应关系定义似然函数 */
    case class SimpleCorrespondenceEvaluator(model: PointDistributionModel[_3D, TriangleMesh],
                                             correspondences: Seq[(PointId, Point[_3D])])
      extends DistributionEvaluator[Sample] {

      override def logValue(sample: Sample): Double = {
        /** 形状和姿态参数的协方差确定的模型实例（原始位置） */
        val currModelInstance = model.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)
        /** 所谓的不确定性就是在计算多元高斯分布时加入误差误差函数noise
         * 这里是在计算多元高斯分布的值，这里假设样本是服从多元高斯的，所以样本的概率密度函数就是
         * 多元高斯分布，所以似然函数就是这些高斯分布的乘积
         * 而使用的数据就是模型到目标面之间的变形*/
        val lmUncertainty = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * sample.parameters.noiseStddev)


        val likelihoods = correspondences.map(correspondence => {
          val (id, targetPoint) = correspondence//对应关系取值
          val modelInstancePoint = currModelInstance.pointSet.point(id)//原始模型的实际物理点（点的信息存储在ID当中现在取出来计算变形）
          val observedDeformation = targetPoint - modelInstancePoint//目标位置物理点-模型实例（原始模型）的物理点
          /** 这里将高斯分布计算的似然函数的值用对数表示 */
          lmUncertainty.logpdf(observedDeformation.toBreezeVector)
        })
        /** 由于值的形式是对数，所以这里是累加 */
        val loglikelihood = likelihoods.sum
        loglikelihood
      }
    }
    /** 由于上述对模型的计算效率很低，为了提高性能，首先应该将模型边缘化到感兴趣的点
     * 但是边缘化会改变点的ID，为了不丢失点的信息需要记录原始ID到新ID的映射这就是marginalizeModelForCorrespondences
     * 的功能*/
    def marginalizeModelForCorrespondences(model: PointDistributionModel[_3D, TriangleMesh],
                                           correspondences: Seq[(PointId, Point[_3D])])
    : (PointDistributionModel[_3D, UnstructuredPointsDomain],
      Seq[(PointId, Point[_3D])]) = {

      val (modelIds, _) = correspondences.unzip
      val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
      val newCorrespondences = correspondences.map(idWithTargetPoint => {
        val (id, targetPoint) = idWithTargetPoint
        val modelPoint = model.reference.pointSet.point(id)
        val newId = marginalizedModel.reference.pointSet.findClosestPoint(modelPoint).id
        (newId, targetPoint)
      })
      (marginalizedModel, newCorrespondences)
    }

    case class CorrespondenceEvaluator(model: PointDistributionModel[_3D, TriangleMesh],
                                       correspondences: Seq[(PointId, Point[_3D])])
      extends DistributionEvaluator[Sample] {
      /** 边缘化模型实例 */
      val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

      override def logValue(sample: Sample): Double = {

        val lmUncertainty = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * sample.parameters.noiseStddev)
        /** 形状和姿态参数的协方差确定的模型实例（原始位置）  */
        val currModelInstance = marginalizedModel
          .instance(sample.parameters.modelCoefficients)
          .transform(sample.poseTransformation)
        /** 重新计算似然函数，不过这次计算的边缘模型 */
        val likelihoods = newCorrespondences.map(correspondence => {
          val (id, targetPoint) = correspondence
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint

          lmUncertainty.logpdf(observedDeformation.toBreezeVector)
        })

        val loglikelihood = likelihoods.sum
        loglikelihood
      }
    }
    /** 缓冲器，记录计算过程中的参数和结果。
     * 为了提高效率，当第二次使用形同的参数是就会直接从缓冲器中直接提取减少计算 */
    case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends DistributionEvaluator[A] {
      val memoizedLogValue = Memoize(evaluator.logValue, 10)

      override def logValue(sample: A): Double = {
        memoizedLogValue(sample)
      }
    }
    /** 似然评估、先验评估、后验评估 */
    val likelihoodEvaluator = CachedEvaluator(CorrespondenceEvaluator(model, correspondences))
    val priorEvaluator = CachedEvaluator(PriorEvaluator(model))

    val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)
    /** 提案生成器构建 */
    /** 形状更新提议 */
    case class ShapeUpdateProposal(paramVectorSize: Int, stddev: Double)
      extends ProposalGenerator[Sample]
        with TransitionProbability[Sample] {

      val perturbationDistr = new MultivariateNormalDistribution(
        DenseVector.zeros(paramVectorSize),
        DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev
      )

      override def propose(sample: Sample): Sample = {
        val perturbation = perturbationDistr.sample()
        val newParameters =
          sample.parameters.copy(modelCoefficients = sample.parameters.modelCoefficients + perturbationDistr.sample)
        sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
      }

      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
        perturbationDistr.logpdf(residual)
      }
    }
    /** 旋转参数更新 */
    case class RotationUpdateProposal(stddev: Double)
      extends ProposalGenerator[Sample]
        with TransitionProbability[Sample] {
      val perturbationDistr =
        new MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * stddev * stddev)
      def propose(sample: Sample): Sample = {
        val perturbation = perturbationDistr.sample
        val newRotationParameters = (
          sample.parameters.rotationParameters._1 + perturbation(0),
          sample.parameters.rotationParameters._2 + perturbation(1),
          sample.parameters.rotationParameters._3 + perturbation(2)
        )
        val newParameters = sample.parameters.copy(rotationParameters = newRotationParameters)
        sample.copy(generatedBy = s"RotationUpdateProposal ($stddev)", parameters = newParameters)
      }
      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = DenseVector(
          to.parameters.rotationParameters._1 - from.parameters.rotationParameters._1,
          to.parameters.rotationParameters._2 - from.parameters.rotationParameters._2,
          to.parameters.rotationParameters._3 - from.parameters.rotationParameters._3
        )
        perturbationDistr.logpdf(residual)
      }
    }
    /** 平移参数更新 */
    case class TranslationUpdateProposal(stddev: Double)
      extends ProposalGenerator[Sample]
        with TransitionProbability[Sample] {

      val perturbationDistr =
        new MultivariateNormalDistribution(DenseVector.zeros(3), DenseMatrix.eye[Double](3) * stddev * stddev)

      def propose(sample: Sample): Sample = {
        val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector.fromBreezeVector(
          perturbationDistr.sample()
        )
        val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
        sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
      }

      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = to.parameters.translationParameters - from.parameters.translationParameters
        perturbationDistr.logpdf(residual.toBreezeVector)
      }
    }
    /** 噪声参数更新 */
    case class NoiseStddevUpdateProposal(stddev: Double)(implicit rng : scalismo.utils.Random)
      extends ProposalGenerator[Sample]
        with TransitionProbability[Sample] {

      val perturbationDistr = breeze.stats.distributions.Gaussian(0, stddev)(rng.breezeRandBasis)

      def propose(sample: Sample): Sample = {
        val newSigma = sample.parameters.noiseStddev +  perturbationDistr.sample()
        val newParameters = sample.parameters.copy(noiseStddev = newSigma)
        sample.copy(generatedBy = s"NoiseStddevUpdateProposal ($stddev)", parameters = newParameters)
      }

      override def logTransitionProbability(from: Sample, to: Sample) = {
        val residual = to.parameters.noiseStddev - from.parameters.noiseStddev
        perturbationDistr.logPdf(residual)
      }
    }
    /** 将三个混合起来只不过频率不同，构成提案生成器 */
    val shapeUpdateProposal = ShapeUpdateProposal(model.rank, 0.1)
    val rotationUpdateProposal = RotationUpdateProposal(0.01)
    val translationUpdateProposal = TranslationUpdateProposal(1.0)
    val noiseStddevUpdateProposal = NoiseStddevUpdateProposal(0.1)

    val generator = MixtureProposal.fromProposalsWithTransition(
      (0.5, shapeUpdateProposal),
      (0.2, rotationUpdateProposal),
      (0.2, translationUpdateProposal),
      (0.1, noiseStddevUpdateProposal)
    )
    /** 定义记录器计算各个生成器的接受拒绝比 */
    class Logger extends AcceptRejectLogger[Sample] {
      private val numAccepted = collection.mutable.Map[String, Int]()
      private val numRejected = collection.mutable.Map[String, Int]()

      override def accept(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]): Unit = {
        val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
        numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
      }

      override def reject(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]): Unit = {
        val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
        numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
      }

      def acceptanceRatios(): Map[String, Double] = {
        val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
        val acceptanceRatios = for (generatorName <- generatorNames) yield {
          val total = (numAccepted.getOrElse(generatorName, 0)
            + numRejected.getOrElse(generatorName, 0)).toDouble
          (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
        }
        acceptanceRatios.toMap
      }
    }
    /** 创建初始样本，就是迭代的初始值 */
    def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
      val normFactor = 1.0 / mesh.pointSet.numberOfPoints
      mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
    }

    val initialParameters = Parameters(
      translationParameters = EuclideanVector(0, 0, 0),
      rotationParameters = (0.0, 0.0, 0.0),
      modelCoefficients = DenseVector.zeros[Double](model.rank),
      noiseStddev = 1.0
    )

    val initialSample = Sample("initial", initialParameters, computeCenterOfMass(model.mean))
    /** 建立马尔科夫链 */
    val chain = MetropolisHastings(generator, posteriorEvaluator)
    val logger = new Logger()
    val mhIterator = chain.iterator(initialSample, logger)
    /** 可视化后验的一些样本 */
    val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 500 == 0) {
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients
        modelView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
      }
      sample
    }
    /** 丢弃一些老样本 */
    val samples = samplingIterator.drop(1000).take(10000).toIndexedSeq

    println(logger.acceptanceRatios())
    // Map(RotationUpdateProposal (0.01) -> 0.27049910873440286, TranlationUpdateProposal (1.0) -> 0.10995475113122172, ShapeUpdateProposal (0.1) -> 0.4661405762525234, NoiseStddevUpdateProposal (0.1) -> 0.8394160583941606)
    /** 选择可能性最大的样本 */
    val bestSample = samples.maxBy(posteriorEvaluator.logValue)
    val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    val resultGroup = ui.createGroup("result")
    ui.show(resultGroup, bestFit, "best fit")


    def computeMean(model: PointDistributionModel[_3D, UnstructuredPointsDomain], id: PointId): Point[_3D] = {
      var mean = EuclideanVector(0, 0, 0)
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        mean += pointForInstance.toVector
      }
      (mean * 1.0 / samples.size).toPoint
    }

    def computeCovarianceFromSamples(model: PointDistributionModel[_3D, UnstructuredPointsDomain],
                                     id: PointId,
                                     mean: Point[_3D]): SquareMatrix[_3D] = {
      var cov = SquareMatrix.zeros[_3D]
      for (sample <- samples) yield {
        val modelInstance = model.instance(sample.parameters.modelCoefficients)
        val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet.point(id)
        val v = pointForInstance - mean
        cov += v.outer(v)
      }
      cov * (1.0 / samples.size)
    }

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    for ((id, _) <- newCorrespondences) {
      val meanPointPosition = computeMean(marginalizedModel, id)
      println(s"expected position for point at id $id  = $meanPointPosition")
      val cov = computeCovarianceFromSamples(marginalizedModel, id, meanPointPosition)
      println(
        s"posterior variance computed  for point at id (shape and pose) $id  = ${cov(0, 0)}, ${cov(1, 1)}, ${cov(2, 2)}"
      )
    }

    /** end  */
  }
}