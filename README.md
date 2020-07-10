# Distance2Pre
Code for my PAKDD-2019. It is implemented by Python27 and Theano.

Distance2Pre: Personalized Spatial Preference for Next Point-of-Interest Prediction

BibTex:

@inproceedings{cui2019distance2pre,

  title={Distance2Pre: Personalized Spatial Preference for Next Point-of-Interest Prediction},
  
  author={Cui, Qiang and Tang, Yuyuan and Wu, Shu and Wang, Liang},
  
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  
  pages={289--301},
  
  year={2019},
  
  organization={Springer}
}

The annotations are in Chinese. Please note that we do not know any information during the test process.

If you have other questions or confusions, please send email to [cuiqiang1990@hotmail.com].

数据集路径：Distance2Pre/code/programs/poidata/

theano就大体看看吧，不用深究，我也不用这个了。

RNN的具体input什么。（1）t 时刻的poi embedding，20维。（2）t-1时刻到 t 时刻两个poi之间的距离间隔，转换为interval，并用20维的embedding表示。

输出预测。（1）训练结束，测试得结果之前，先再用用户训练序列重新走一下RNN，得到训练序列最后一个时刻的用户hidden state作为user vector。（2）用user vector * POI vector得到对所有poi的偏好得分。（3）用user vector得到对下次各个距离间隔不同偏好的得分st，就是一个interval预测出来一个得分。（4）每个训练序列的最后一个poi，计算它和所有poi的距离间隔并转换为interval。这一步在训模型之前做完，存起来，测试时直接取出来用。再根据刚计算出来的st把这些interval转换为距离偏好得分。（5）两个偏好分数都有了，再用模型里提到的线性、非线性融合方式得到总得分。

模型测试时，每个用户的test list就一个真值poi，并且预测时没有任何关于该真值poi的信息（比如空间位置等）。因此如果测试时已经有了真值poi的空间位置（有些文章这么做的），我的模型就不适用了。
