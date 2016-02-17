import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four
 * methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	// class attributes
	private double p_of_spam;
	private double p_of_ham;
	// conditional probability P(word | label)
	private double p_given_spam_denominator;
	private double p_given_ham_denominator;

	// keys are words
	private Map<String, Integer> count_given_spam;
	private Map<String, Integer> count_given_ham;
	private Map<String, Double> p_given_spam;
	private Map<String, Double> p_given_ham;

	private List<String> vocab;

	// for smoothifying
	private final double delta = 0.00001;

	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
	public void train(Instance[] trainingData, int v) {
		// Implement

		// Initialize all that you want
		this.vocab = new ArrayList<String>();
		int num_instances = trainingData.length;
		// spam count and ham count
		int[] label_count = new int[2];
		// class-conditional counts
		this.count_given_spam = new HashMap<String, Integer>(v);
		this.count_given_ham = new HashMap<String, Integer>(v);

		for (Instance instance : trainingData) {
			Label curr_label = instance.label;
			// update the counts appropriately
			if (curr_label == Label.SPAM) {
				label_count[0]++;
			} else if (curr_label == Label.HAM) {
				label_count[1]++;
			}

			for (String word : instance.words) {
				// Add to the vocabulary if we haven't seen the word yet
				if (!this.vocab.contains(word)) {
					this.vocab.add(word);
				}

				if (curr_label == Label.SPAM) {
					// DefaultDict of python would have meant 1 line of code ;)
					if (!this.count_given_spam.containsKey(word)) {
						this.count_given_spam.put(word, new Integer(1));
					} else {
						int newVal = this.count_given_spam.get(word) + 1;
						this.count_given_spam.put(word, newVal);
					}
				} else if (curr_label == Label.HAM) {
					if (!this.count_given_ham.containsKey(word)) {
						this.count_given_ham.put(word, new Integer(1));
					} else {
						int newVal = this.count_given_ham.get(word) + 1;
						this.count_given_ham.put(word, newVal);
					}
				}
			}
		}

		// Calculate the probability
		this.p_of_spam = (double) (label_count[0]) / (double) (num_instances);
		this.p_of_ham = (double) (label_count[1]) / (double) (num_instances);

		this.p_given_spam = new HashMap<String, Double>(v);
		this.p_given_ham = new HashMap<String, Double>(v);

		// smoothify
		for (String word : this.vocab) {
			double num = this.delta;
			if (this.count_given_spam.containsKey(word)) {
				num += (double) (this.count_given_spam.get(word));
			}

			double denominator = (double) (v) * this.delta;
			for (String token : this.vocab) {
				if (this.count_given_spam.containsKey(token)) {
					denominator += (double) (this.count_given_spam.get(token));
				}
			}

			this.p_given_spam.put(word, (double) (num / denominator));
			this.p_given_spam_denominator = denominator;
		}

		for (String word : this.vocab) {
			double num = this.delta;
			if (this.count_given_ham.containsKey(word)) {
				num += (double) (this.count_given_ham.get(word));
			}

			double denominator = (double) (v) * this.delta;
			for (String token : this.vocab) {
				if (this.count_given_ham.containsKey(token)) {
					denominator += (double) (this.count_given_ham.get(token));
				}
			}

			this.p_given_ham.put(word, (double) (num / denominator));
			this.p_given_ham_denominator = denominator;
		}
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or
	 * P(HAM)
	 */
	@Override
	public double p_l(Label label) {
		// Implement
		if (label == Label.SPAM) {
			return this.p_of_spam;
		} else {
			return this.p_of_ham;
		}
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		// Implement
		if (!this.vocab.contains(word)) {
			if (label == Label.SPAM) {
				return this.delta / this.p_given_spam_denominator;
			} else {
				return this.delta / this.p_given_ham_denominator;
			}
		}

		if (label == Label.SPAM) {
			return this.p_given_spam.get(word);
		} else {
			return this.p_given_ham.get(word);
		}
	}

	/**
	 * Classifies an array of words as either SPAM or HAM.
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		// Implement

		ClassifyResult result = new ClassifyResult();
		result.log_prob_spam = Math.log(p_of_spam);
		result.log_prob_ham = Math.log(p_of_ham);

		for (String word : words) {
			result.log_prob_spam += Math.log(p_w_given_l(word, Label.SPAM));
			result.log_prob_ham += Math.log(p_w_given_l(word, Label.HAM));
		}

		// Naive Bayes' Classify
		if (result.log_prob_spam > result.log_prob_ham) {
			result.label = Label.SPAM;
		} else if (result.log_prob_ham > result.log_prob_spam) {
			result.label = Label.HAM;
		} else if (this.p_of_spam > this.p_of_ham) {
			result.label = Label.SPAM;
		} else {
			result.label = Label.HAM;
		}
		return result;
	}
}
