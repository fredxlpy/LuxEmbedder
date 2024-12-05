# LuxEmbedder
This is the repository for [*LuxEmbedder: A Cross-Lingual Approach to Enhanced Luxembourgish Sentence Embeddings* (Philippy et al., 2024)](https://doi.org/10.48550/arXiv.2412.03331), accepted at [COLING 2025](https://coling2025.org). The paper introduces ***LuxEmbedder***, a cross-lingual Luxembourgish sentence embedding model, as well as ***LuxAlign***, a parallel dataset for Luxembourgish-French and Luxembourgish-English, and ***ParaLux***, a benchmark for Luxembourgish paraphrase detection.

## 🤖 LuxEmbedder
Download the pre-trained LuxEmbedder model [here](https://huggingface.co/fredxlpy/LuxEmbedder).

## 📂 LuxAlign
The parallel LB-EN & LB-FR data that was used to train LuxEmbedder can be downloaded [here](https://huggingface.co/datasets/fredxlpy/LuxAlign).

#### Examples:
| Luxembourgish Sentence                                                                                           | English/French Sentence                                                                             |
|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| D’Police sicht no engem Mann, deen an der Stad mat enger geklauter Kreditkaart Suen opgehuewen huet.             | The police is looking for a man who withdrew money with a stolen credit card in Luxembourg City.   |
| D’Temperaturen am Grand-Duché sinn an der Moyenne em 1.3 Grad an d’Luucht gaangen.                               | Temperatures in the Grand Duchy have risen by 1.3 degrees on average.                              |
| Déi Petitioun ass vun 336.000 Persounen aus 112 Länner ënnerschriwwe ginn.                                       | Cette pétition a été signée par 336.000 personnes originaires de 112 pays.                        |
| Am September 2013 hat fir d’éischte Kéier e Lëtzebuerger den Jackpot gewonnen.                                   | En septembre 2013, un Luxembourgeois avait pour la 1e fois remporté le jackpot.                   |


## 📊 ParaLux
ParaLux is a Luxembourgish paraphrase detection benchmark and can be downloaded [here](https://huggingface.co/datasets/fredxlpy/ParaLux).

#### Examples:
| Anchor Sentence                                                                                     | Paraphrase                                                                                 | Not Paraphrase                                                                         |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **Mexiko gewënnt 3-1 géint Kroatien.** <br><sub>*Mexico wins 3-1 against Croatia.*</sub>            | **Kroatien verléiert 1-3 géint Mexiko.** <br><sub>*Croatia loses 3-1 against Mexico.*</sub> | **Kroatien gewënnt 3-1 géint Mexiko.** <br><sub>*Croatia wins 3-1 against Mexico.*</sub> |
| **De Sträit tëscht Süd- a Nordkorea spëtzt sech weider zou.** <br><sub>*The dispute between South and North Korea continues to escalate.*</sub> | **D’Verhältnis tëscht Nord- a Südkorea gëtt ëmmer méi schlecht.** <br><sub>*The relationship between South and North Korea is getting worse and worse.*</sub> | **De Sträit tëscht Süd- a Nordkorea entspaant sech weider.** <br><sub>*The dispute between South and North Korea continues to ease.*</sub> |



### 📜 Citation
If you find this paper and repository useful in your research, please cite:
```bibtex
@misc{philippy2024luxembedder,
      title={LuxEmbedder: A Cross-Lingual Approach to Enhanced Luxembourgish Sentence Embeddings}, 
      author={Fred Philippy and Siwen Guo and Jacques Klein and Tegawendé F. Bissyandé},
      year={2024},
      eprint={2412.03331},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.03331}, 
}
```

### 💬 Contact
For questions or collaboration opportunities, reach out to:
- Fred Philippy: [fred@zortify.com](mailto:fred@zortify.com)

We would like to express our sincere gratitude to RTL Luxembourg for providing the raw seed data that served as the foundation for this research. Those interested in obtaining this data are encouraged to reach out to [RTL Luxembourg](https://www.rtl.lu) or Mr. Tom Weber via [ai@rtl.lu](mailto:ai@rtl.lu).