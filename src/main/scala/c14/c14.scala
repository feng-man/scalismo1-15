package c14

import java.awt.Color

import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import breeze.stats.distributions.Gaussian
import breeze.stats.meanAndVariance

import scalismo.ui.api.ScalismoUI
import breeze.linalg.{DenseMatrix, DenseVector}
object c14 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    //val ui = ScalismoUI()

    /** begin */
    /** begin */
    /** 生成数据 */
    val a = 0.2
    val b = 3
    val sigma2 = 0.5
    val errorDist = breeze.stats.distributions.Gaussian(0, sigma2)
    val data = for (x <- 0 until 100) yield {
      (x.toDouble, a * x + b + errorDist.draw())
    }

    case class Parameters(a : Double, b:  Double, sigma2 : Double)

    case class Sample(parameters : Parameters, generatedBy : String)
    /** Evaluators: Modelling the target density */

    /** 定义似然函数 */
    case class LikelihoodEvaluator(data : Seq[(Double, Double)]) extends DistributionEvaluator[Sample] {

      override def logValue(theta: Sample): Double = {

        val likelihoods = for ((x, y) <- data) yield {
          val likelihood = breeze.stats.distributions.Gaussian(
            theta.parameters.a * x + theta.parameters.b, theta.parameters.sigma2)

          likelihood.logPdf(y)//使用的是对数概率
        }
        likelihoods.sum//由于使用的是对数概率所以这里是求和（一般似然函数是求乘积相互独立，多以乘积是联合概率密度 // ）
      }
    }

    /** 定义先验分布 */
    object PriorEvaluator extends DistributionEvaluator[Sample] {
      /** 定义参数的分布两个是高斯一个是logNormal */
      val priorDistA = breeze.stats.distributions.Gaussian(0, 1)
      val priorDistB = breeze.stats.distributions.Gaussian(0, 10)
      val priorDistSigma = breeze.stats.distributions.LogNormal(0, 0.25)
      /** 将分布的值转换为对数形式 */
      override def logValue(theta: Sample): Double = {
        priorDistA.logPdf(theta.parameters.a)
        + priorDistB.logPdf(theta.parameters.b)
        + priorDistSigma.logPdf(theta.parameters.sigma2)
      }
    }
    /** 由先验分布和似然函数（将试验数据代入似然函数计算）计算后验分布 ，但是这是没有归一化的后验分布*/
    val posteriorEvaluator = ProductEvaluator(PriorEvaluator, LikelihoodEvaluator(data))
    /** 定义ProposalGenerator的特征 */



    /** 定义ProposalGenerator的具体函数 */
    case class RandomWalkProposal(stepLengthA: Double, stepLengthB : Double, stepLengthSigma2 : Double)(implicit rng : scalismo.utils.Random)
      extends ProposalGenerator[Sample] with TransitionProbability[Sample] {
      /** 根据步长更新每个参数的值 */
      override def propose(sample: Sample): Sample = {
        val newParameters = Parameters(
          a = sample.parameters.a + rng.breezeRandBasis.gaussian(0, stepLengthA).draw(),
          b = sample.parameters.b + rng.breezeRandBasis.gaussian(0, stepLengthB).draw(),
          sigma2 = sample.parameters.sigma2 + rng.breezeRandBasis.gaussian(0, stepLengthSigma2).draw(),
        )

        Sample(newParameters, s"randomWalkProposal ($stepLengthA, $stepLengthB)")
      }
      /** 根据新参数的值计算先验分布并转换成对数概率 */
      override def logTransitionProbability(from: Sample, to: Sample) : Double = {

        val stepDistA = breeze.stats.distributions.Gaussian(0, stepLengthA)
        val stepDistB = breeze.stats.distributions.Gaussian(0, stepLengthB)
        val stepDistSigma2 = breeze.stats.distributions.Gaussian(0, stepLengthSigma2)
        val residualA = to.parameters.a - from.parameters.a
        val residualB = to.parameters.b - from.parameters.b
        val residualSigma2 = to.parameters.sigma2 - from.parameters.sigma2
        stepDistA.logPdf(residualA)  + stepDistB.logPdf(residualB) + stepDistSigma2.logPdf(residualSigma2)
      }
    }

    /** 定义游走步长函数 */
    val smallStepProposal = RandomWalkProposal(0.01, 0.01, 0.01)
    val largeStepProposal = RandomWalkProposal(0.1, 0.1, 0.1)
    /** 将两种步长混合起来80%用小步长20%用大步长，从而组成生成器 */
    val generator = MixtureProposal.fromProposalsWithTransition[Sample](
      (0.8, smallStepProposal),
      (0.2, largeStepProposal)
    )

    /** Building the Markov Chain */
    /** 由后验分布和生成器组成马尔科夫 */
    val chain = MetropolisHastings(generator, posteriorEvaluator)
    /** 迭代初始值 */
    val initialSample = Sample(Parameters(0.0, 0.0, 1.0), generatedBy="initial")
    val mhIterator = chain.iterator(initialSample)

    val samples = mhIterator.drop(5000).take(15000).toIndexedSeq
    //drop用法为删除mhIterator的前5000个值，take用法为取前15000个值
    val meanAndVarianceA = meanAndVariance(samples.map(_.parameters.a))
    println(s"Estimates for parameter a: mean = ${meanAndVarianceA.mean}, var = ${meanAndVarianceA.variance}")
    val meanAndVarianceB = meanAndVariance(samples.map(_.parameters.b))
    println(s"Estimates for parameter b: mean = ${meanAndVarianceB.mean}, var = ${meanAndVarianceB.variance}")
    val meanAndVarianceSigma2 = meanAndVariance(samples.map(_.parameters.sigma2))
    println(s"Estimates for parameter sigma2: mean = ${meanAndVarianceSigma2.mean}, var = ${meanAndVarianceSigma2.variance}")

    /** Debugging the Markov chain */

    class Logger extends AcceptRejectLogger[Sample] {
      private val numAccepted = collection.mutable.Map[String, Int]()
      private val numRejected = collection.mutable.Map[String, Int]()

      override def accept(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]
                         ): Unit = {
        val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
        numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
      }

      override def reject(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]
                         ): Unit = {
        val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
        numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
      }


      def acceptanceRatios() : Map[String, Double] = {
        val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
        val acceptanceRatios = for (generatorName <- generatorNames ) yield {
          val total = (numAccepted.getOrElse(generatorName, 0)
            + numRejected.getOrElse(generatorName, 0)).toDouble
          (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
        }
        acceptanceRatios.toMap
      }
    }

    val logger = new Logger()
    val mhIteratorWithLogging = chain.iterator(initialSample, logger)

    val samples2 = mhIteratorWithLogging.drop(5000).take(15000).toIndexedSeq

    println("acceptance ratio is " +logger.acceptanceRatios())
    // acceptance ratio is Map(randomWalkProposal (0.1, 0.1) -> 0.0066699604743083, randomWalkProposal (0.01, 0.01) -> 0.12676321233778445)

    /** end */

    /** end  */
  }
}