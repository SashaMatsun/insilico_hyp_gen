# insilico_hyp_gen

The code implementation of a model for disease hypotheses generation, using a given bioentity knowledge graph. The Levenstein distance between a generated and ground-truth hypotheses is utilized as performance metric. The method consists of weight assignment to the edges using a link prediction model with further shortest path search between the target gene and a disesase. The results for hypothesis generation and link prediction are provided below.

![alt text](https://github.com/SashaMatsun/insilico_hyp_gen/blob/main/images/ins_res.png)
