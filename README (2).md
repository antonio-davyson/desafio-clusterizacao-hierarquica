# Clusterização hierárquica de espécies fictícias

Desafio Rocketseat: agrupar 600 espécies fictícias com clusterização hierárquica, gerar dendrogramas, cortar em pelo menos 10 clusters,
descrever os grupos e analisar as relações entre eles.

## Conteúdo
- `clusterizacao_hierarquica_especies.py` — exploração, preparação, linkages, dendrogramas, corte, perfis e análise (no final do arquivo)
- `species_dataset.csv` — dataset original; `species_com_clusters.csv` — dataset com o cluster de cada espécie
- `exploracao.png`, `dendrogramas.png`, `dendrograma_ward_corte.png`, `perfil_clusters.png` — gráficos

## Como executar
```bash
pip install -r requirements.txt
python clusterizacao_hierarquica_especies.py
```

## Abordagem
Numéricas padronizadas, booleanas 0/1, categóricas em one-hot. Ligações comparadas: Ward, average, complete e single.
Ward foi o modelo principal (clusters mais equilibrados), cortado em 10 clusters.

## Principais achados
- Os eixos que mais organizam a árvore são o número de olhos e o de patas.
- A estrutura de grupos é fraca (silhueta ≈ 0,04 com k=10) e pouco estável entre métodos de ligação, o que é coerente com atributos praticamente independentes.
- O dendrograma mede semelhança entre atributos, não parentesco evolutivo real.
