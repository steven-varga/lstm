
<!-- INTITILIZE PROJECT

git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/steven-varga/lstm-ukf.git
git push -u origin master

-->

Sequential training of LSTM with Unscented Kalman Filter
=========================================================

A comparative study of custom, second order training method with first order, [gradient descent method][1] based on 
[UCI: Human Activity Dataset][2]. 
The [performance is between 91%][1] and [94%][3] was reported by [Guillaume Chevalier][4]. 

For the initial setup Julia, a relatively new player to data-science is chosen, with minimal
dependency.

TODO:
-----

1. intialize project, with references
2. obtain, and import dataset
3. build test-harness
4. LSTM and Kalman Filter code
5. verify for code correctness
6. obtain result
7. compare and discuss with existing results


training method:
----------------
* The Unscented Kalman Filter for Nonlinear Estimation, Eric A. Wan and Rudolph van der Merwe
* THE SQUARE-ROOT UNSCENTED KALMAN FILTER FOR STATE AND PARAMETER-ESTIMATION, Rudolph van der Merwe and Eric A. Wan, 
* Low Rank Updates for the Cholesky Decomposition, Matthias Seeger, 2008 
* A New Extension of the Kalman Filter to Nonlinear Systems, Simon J. Julier Jeffrey K. Uhlmann
* Kalman Filtering And Neural Networks,  Simon Haykin

model:
------
* LSTM: A Search Space Odyssey Klaus Greff, Rupesh Kumar Srivastava, Jan Koutnı́k, Bas R. Steunebrink, Jürgen Schmidhuber
* An Empirical Exploration of Recurrent Network Architectures Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever
* Highway Networks Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
* Depth-Gated Recurrent Neural Networks Kaisheng Yao, Trevor Cohn, Katerina Vylomova, Kevin Duh, Chris Dyer


UCI dataset:
------------------------
* Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012 
* Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz. Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic. Journal of Universal Computer Science. Special Issue in Ambient Assisted Living: Home Care. Volume 19, Issue 9. May 2013
* Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. 4th International Workshop of Ambient Assited Living, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings. Lecture Notes in Computer Science 2012, pp 216-223.
* Jorge Luis Reyes-Ortiz, Alessandro Ghio, Xavier Parra-Llanas, Davide Anguita, Joan Cabestany, Andreu Catal?. Human Activity and Motion Disorder Recognition: Towards Smarter Interactive Cognitive Environments. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013
* Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013

[1]: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
[2]: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
[3]: https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
[4]: https://github.com/guillaume-chevalier

