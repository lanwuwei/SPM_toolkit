## DecAtt
DecAtt represents decomosable attention model, which was first proposed in this [paper](https://arxiv.org/pdf/1606.01933v1.pdf):
	
	@inproceedings{parikh-EtAl:2016:EMNLP2016,
	  author     = {Parikh, Ankur  and  T\"{a}ckstr\"{o}m, Oscar  and  Das, Dipanjan  and  Uszkoreit, Jakob},
  	  title    = {A Decomposable Attention Model for Natural Language Inference},
  	booktitle  = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  	  year     = {2016}
  	} 
  
## A few notes
1. Creat data folder and download SNLI, MNLI and Quora [here](https://drive.google.com/drive/folders/1h4PnoST3MdqEdRfRKpBe-JlxzKIYvv-F?usp=sharing), then change the corresponding path for data and embedding vectors in main file.
2. If you want to run DecAtt on SNLI dataset, just type: python main_snli.py
