# Trabalho 1 - Algoritmos de Busca

Integrantes:
- José Vinícius Evangelista Dias de Souza - 537071
- Francisco Paulino Arruda Filho  - 538451
- Carlos Ryan Santos Silva - 473007

## Introdução

Neste trabalho, implementamos uma solução para o quebra-cabeça 8-puzzle, que consiste em um tabuleiro 3x3 com 8 peças numeradas de 1 a 8 e um espaço vazio. O objetivo é mover as peças para que elas fiquem ordenadas de forma crescente desconsiderando o espaço vazio, ou seja, há 9 soluções possíveis, uma para cada possibilidade do espaço vazio. O quebra-cabeça é resolvido utilizando os seguintes algoritmos de busca:

- Busca em Largura (BFS)
- Busca em Profundidade (DFS)
- Busca de Custo Uniforme (Uniform Cost Search) ou Dijkstra
- Busca Gulosa (Greedy Search)
- Busca A* (A* Search)

Em seguida executamos os experimentos definidos no trabalho, coletando dados sobre:
- Tempo de execução
- Número de nós gerados
- Número de nós visitados
- Custo total (Baseado nas funções de custo c1, c2 , c3 e c4)
- Profundidade da solução

## Preparação do Ambiente e Execução

Primeiro vamos preparar o ambinte para a execução do trabalho, instalando as dependências necessárias e executando os testes para garantir que tudo esteja funcionando corretamente.

### Colab

Para executar o trabalho no colab, `basta abrir o link do notebook no Google Colab e executar as células sequencialmente`. O notebook já contém as dependências necessárias instaladas, portanto não é necessário instalar nada adicionalmente.

No entanto, não é necessário executar o notebook se quiser apenas visualizar o código e ver os resultados dos experimentos. Nós já executamos o notebook e os resultados já estão exibidos como saída das células, ao executar novamente a saída da celula será limpa e os resultados serão recalculados.

❗ Não recomendamos execute o notebook pois o **tempo de execução é longo** e pode levar **mais de 1 hora** para ser concluído.

### Local
Para executar o trabalho, você pode utilizar o ambiente local ou um ambiente virtual. Recomendamos o uso de um ambiente virtual para isolar as dependências do projeto.

Dependências:
  - [Python >= 3.11.9](https://www.python.org/downloads/release/python-3119/)
  - [Numpy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [Plotly](https://plotly.com/python/)
  - [Ipykernel](https://ipykernel.readthedocs.io/en/stable/)


Para criar um ambiente virtual, execute os seguintes comandos no terminal:

```bash
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate
```

Em seguida, instale as dependências necessárias:

```bash
pip install numpy pandas plotly ipykernel
```

Depois é só executar o notebook no Jupyter Notebook ou Jupyter Lab ou no Visual Studio Code.

### Importando as bibliotecas necessárias

Utilizamos em maioria as bibliotecas padrão do Python, como:
- `collections` para manipulação de filas e pilhas;
- `typing` para anotações de tipos;
- `time` para medir o tempo de execução;
- `heapq` para implementar filas de prioridade;
- `pickle` para serialização de objetos.
- `itertools` para manipulação de iteradores e combinações.
- `enum` para definir enumeradores.
- `multiprocessing` para a execução paralela de algoritmos.
- `dataclasses` para definir classes de dados imutáveis.
- `abc` para definir classes abstratas.

Além disso, utilizamos as seguintes bibliotecas de terceiros:
- `numpy` para manipulação de arrays e matrizes;
- `pandas` para manipulação de dados tabulares;
- `plotly` para visualização de dados interativos;
- `ipykernel` para executar o notebook no Jupyter Notebook ou Jupyter Lab.
- `tqdm` para exibir barras de progresso durante a execução dos algoritmos.

### Action

A classe `Action` é uma representação de uma ação que pode ser executada em um ambiente. Ela é um `IntEnum` que define as ações disponíveis, no caso:

- `UP = 1`: Mover para cima
- `DOWN = 2`: Mover para baixo
- `LEFT = 3`: Mover para a esquerda
- `RIGHT = 4`: Mover para a direita

A escolha do `IntEnum` permite que as ações sejam tratadas de forma mais segura e legível, evitando erros de digitação e melhorando a clareza do código e a documentação.

Metódos:
- `opposite`: Método estático que Retorna a ação oposta da ação atual. Por exemplo, se a ação for `UP`, a ação oposta será `DOWN`. Isso é útil para reverter ações em algoritmos de busca ou para evitar movimentos redundantes.
- `inverse`: Propriedade da instância que retorna a ação oposta da ação atual, utilizando o método `opposite`. Isso permite acessar a ação oposta de forma mais intuitiva e direta.


## Modelagem e implementação

Nesta seção, vamos descrever a modelagem e implementação do quebra-cabeça 8-puzzle, incluindo a definição do problema, a classe `Problem`, a classe `State`, a classe `Frontier` e as implementações concretas de filas.

### Problem

A classe `Problem` é uma representação de um problema que pode ser resolvido por um agente. No caso, ela representa especificamente o 8-puzzle, que é um quebra-cabeça onde o objetivo é mover os blocos para alcançar uma configuração específica.

---

#### Imutabilidade

Modelamos o problema como uma classe imutável, o que significa que uma vez criado, o estado do problema não pode ser alterado. Isso é importante para garantir que o agente possa explorar o espaço de estados de forma consistente e previsível. Qualquer modificação no estado do problema resulta na criação de uma nova instância da classe `Problem`, preservando o estado original.

Facilita a implementação de algoritmos de busca, pois os estados não mudam inesperadamente durante a execução do algoritmo. Isso também permite que o agente mantenha um histórico claro das ações realizadas e dos estados visitados, facilitando a depuração e a análise do comportamento do agente.

---

#### Construtor

O construtor da classe `Problem` recebe opcionalmente um estado inicial do quebra-cabeça como um `np.ndarray` de forma (3, 3) e opcionalmente um log de ações como uma `tuple[Action]`. Se o estado inicial não for fornecido, ele é gerado aleatoriamente. O estado objetivo padrão é a configuração resolvida do quebra-cabeça.

Outra forma de criar uma instância de `Problem` é através do método de classe `make_solvable` que recebe um interiro para ser usado como semente aleatória `random_state` e um número de permutações `n_permutations`, gerando um estado inicial aleatório que tem N permutações aleatórias, garantindo que o quebra-cabeça seja solucionável. Isso é útil para criar instâncias de problemas que são desafiadoras, mas ainda assim resolvíveis.

---

#### Atributos

- `state`: O estado atual do quebra-cabeça, representado por um `np.ndarray` de forma (3, 3), onde cada elemento é um inteiro representando o número do bloco ou `zero` para o espaço vazio. Sempre retorna uma cópia do estado atual para evitar modificações indesejadas.
- `log`: uma `tuple[Action]` que armazena as ações realizadas pelo agente para alcançar o estado atual.
- `int_log`: um `Inp.ndarray[tuple[int], np.dtype[np.int64]]`= que retorna o log de ações em forma de `np.ndarray` de inteiros, onde cada ação é representada por seu valor inteiro correspondente. Isso é útil para algoritmos que precisam processar o log de ações de forma eficiente.
- `is_solved`: um booleano que indica se o problema foi resolvido, ou seja, se o estado atual é igual ao estado objetivo, que no caso do 8-puzzle é a configuração onde os blocos estão ordenados de 1 a 8, com o espaço vazio representado por 0 em qualquer posição.
- `all_actions`: um dicionário `dict[Action, Problem | None]` que mapeia cada ação para uma um estado resultante (se houver), permitindo que o agente saiba quais ações pode realizar a partir do estado atual.
- `possible_actions`: um  dicionário `dict[Action, Problem]`  que mapeia cada ação para um novo estado resultante filtrado, ou seja, apenas as ações que podem ser realizadas a partir do estado atual.
- `neighbors`: uma lista de tuplas `list[tuple[Action, Problem]]` que contém pares de estados vizinhos e as ações que levam a esses estados. filtrando apenas as ações que resultam em estados válidos (que não são nulos e não revertem a ação anterior para evitar ciclos).
- `last_action`: a última ação realizada pelo agente, que é usada para evitar ações inválidas (como mover o espaço vazio de volta para a posição anterior).
- `zero_position`: uma tupla `tuple[int, int]` que representa a posição do espaço vazio no estado atual do quebra-cabeça.

- Ações:
    1. `action_up`:  Retorna `Problem | None` uma nova instância de `Problem` com o espaço vazio movido para cima, se possível.
    2. `action_down`: Retorna `Problem | None` uma nova instância de `Problem` com o espaço vazio movido para baixo, se possível.
    3. `action_left`: Retorna `Problem | None` uma nova instância de `Problem` com o espaço vazio movido para a esquerda, se possível.
    4. `action_right`: Retorna `Problem | None` uma nova instância de `Problem` com o espaço vazio movido para a direita, se possível.

    Nos casos em que a ação não puder ser realizada (por exemplo, se o espaço vazio já estiver na borda do quebra-cabeça), o método retorna `None`.

---

#### Métodos

- `make_solvable`: Método de classe que recebe um inteiro `random_state` para ser usado como semente aleatória e um número de permutações `n_permutations`, gerando um estado inicial aleatório que tem N permutações aleatórias, garantindo que o quebra-cabeça seja solucionável. Retorna uma nova instância de `Problem` com o estado inicial gerado.
- `execute_log`: Recebe uma `Sequence[Action]` e executa as ações na ordem especificada, atualizando o estado do problema e o log de ações. Retorna `tuple["Problem", list["Problem"]]`, que contém o novo estado do problema e uma lista de estados intermediários resultantes após cada ação.
- `__str__`: Retorna uma representação em string do estado atual do problema, formatada como um quebra-cabeça 3x3 e o log de ações realizado até o momento.
- `__eq__`: Compara dois objetos `Problem` para verificar se eles são iguais, no caso, se *os estados atuais* são iguais, ignorando o log de ações. Retorna um booleano indicando se os objetos são iguais. Essa escolha permite que o agente compare estados de forma eficiente e clara, sem se preocupar com o histórico de ações facilitando a implementação dos algoritmos.
- `__hash__`: Retorna o hash do **estado atual** do problema, permitindo que ele seja usado como chave em dicionários e conjuntos. Isso é útil para algoritmos que precisam armazenar estados visitados ou explorar o espaço de estados de forma eficiente.
- `__repr__`: Retorna uma representação em string do objeto `Problem`, é a mesma que o método `__str__`, mas é usada para depuração e exibição de informações do objeto.

### Frontier

Os algoritmos de busca em grafo utilizam uma fila para armazenar os estados a serem explorados da fronteira. Dependendo do algoritmo, essa fila pode ser uma pilha (LIFO) ou uma fila (FIFO) ou uma fila de prioridade. A classe `Frontier` é uma classe abstrata genérica que define a interface para essas filas, permitindo que diferentes implementações sejam criadas para diferentes algoritmos de busca.

A interface define os seguintes métodos:
- `put`: Adiciona um estado à fila.
- `pop`: Remove e retorna o próximo estado a ser explorado.
- `is_empty`: Verifica se a fila está vazia.
- `__len__`: Retorna o número de estados na fila.
- `__contains__`: Verifica se um estado está na fila, permitindo que o agente verifique rapidamente se um estado já foi visitado ou está na fila para exploração futura.
- `__iter__`: Retorna um iterador sobre os estados na fila, permitindo que o agente percorra os estados na fila de forma eficiente e flexível.

Além disso, a classe `Frontier` é genérica, permitindo que diferentes tipos de filas sejam criados com base em diferentes tipos de estados. Isso permite que o agente utilize a mesma interface para diferentes implementações de filas, tornando o código mais flexível e extensível.

A classe também define dois métodos estáticos da classe:
- `from_iterable`: Cria uma instância da fila a partir de um iterável de estados, permitindo que o agente inicialize a fila com um conjunto de estados pré-existentes, ela deve ser usada pelas implementações concretas da classe `Frontier` para criar uma fila a partir de um iterável de estados. Ao tentar criar uma instância de `Frontier` diretamente, o método `from_iterable` lança um erro `TypeError`, pois a classe `Frontier` é abstrata e não pode ser instanciada diretamente.
- `from_strategy`: Cria uma instância da fila a partir de um modo de fila específico, os seguintes modos são suportados:
  - `queue`: Fila FIFO (First In, First Out), onde os estados são explorados na ordem em que foram adicionados.
  - `stack`: Fila LIFO (Last In, First Out), ou pilha, onde os estados mais recentemente adicionados são explorados primeiro.
  - `priority`: Fila de prioridade, onde os estados são explorados com base em um critério de prioridade definido por uma função de custo.
  Esse método permite instanciar a fila de acordo com o modo desejado sem expor a implementação concreta da fila, tornando o código mais flexível e extensível. Respeitando o princípio de inversão de dependência, o agente não precisa conhecer a implementação concreta da fila, apenas a interface definida pela classe `Frontier`.

---

#### Implementações Concretas

- `FIFOFrontier`: Implementa uma fila FIFO (First In, First Out), onde os estados são explorados na ordem em que foram adicionados. Utiliza uma `deque` para armazenar os estados internos, permitindo operações eficientes de adição e remoção. O método `append` adiciona novos estados ao final da fila, enquanto o método `popleft` remove e retorna o primeiro estado da fila, garantindo que os estados sejam explorados na ordem correta.

- `LIFOFrontier`: Implementa uma fila LIFO (Last In, First Out), ou pilha, onde os estados mais recentemente adicionados são explorados primeiro. Utiliza uma `list` para armazenar os estados internos e o método `append` para adicionar novos estados ao final da fila. O método `pop` remove e retorna o último estado da fila, garantindo que os estados sejam explorados na ordem correta.

- `PriorityFrontier`: Implementa uma fila de prioridade, onde os estados são explorados com base em um critério de prioridade definido por uma função de custo. Utiliza a biblioteca `heapq` para manter a ordem dos estados com base em seus custos. O método `put` adiciona novos estados à fila com um custo associado, enquanto o método `pop` remove e retorna o estado com o menor custo, garantindo que os estados sejam explorados na ordem correta.

---

A ideia é ter uma interface comum para todas as filas, permitindo uma abordagem flexível e extensível para diferentes algoritmos de busca. Cada implementação concreta pode ser usada conforme necessário, dependendo do algoritmo de busca escolhido pelo agente.

### Solver

A classe `Solver` é responsável por resolver o problema do quebra-cabeça 8-puzzle utilizando diferentes algoritmos de busca. Essa classe é a parametrizada com os seguintes parametros:

- `heuristic`: Uma função heurística que calcula o custo estimado de alcançar o estado objetivo a partir de um estado atual. Essa função é usada pelos algoritmos de busca gulosa e A* para guiar a busca em direção à solução.
- `cost_function`: Uma função de custo que calcula o custo real de alcançar um estado a partir do estado inicial. Essa função é usada pelo algoritmo de busca de custo uniforme para garantir que os estados sejam explorados na ordem correta com base no custo real.
- `queue_mode`: Uma modo de fila, usado pela classe `Frontier` para determinar qual implementação de fila deve ser usada para armazenar os estados a serem explorados. Os modos suportados são: `queue` (FIFO), `stack` (LIFO) e `priority` (fila de prioridade).
- `best_first`: Um booleano que indica se o algoritmo de busca deve ser executado em modo "best first", ou seja, sse deve ou não gerar nós filhos para estados já visitados com custo menor em relação ao estado atual. Isso é usado em algoritmos como A* e busca gulosa e Dijkstra para evitar a exploração de estados já visitados com custo maior, melhorando a eficiência da busca e evitando ciclos desnecessários.

A ideia é que cada algoritmo de busca pode ser instanciado com diferentes combinações de heurísticas, funções de custo e modos de fila, permitindo que uma única classe `Solver`  seja usada para resolver o problema baseado em várias estratégias de busca.

---

#### Métodos

- `from_algorithm`: Método de classe que recebe um algoritmo de busca e retorna uma instância da classe `Solver` com os parâmetros apropriados para o algoritmo. Os algoritmos suportados são: `bfs` (Busca em Largura), `dfs` (Busca em Profundidade), `dijkstra` (Busca de Custo Uniforme), `greedy` (Busca Gulosa) e `a_star` (Busca A*). Esse método permite criar uma instância do solucionador com base no algoritmo desejado, facilitando a configuração e execução do agente.
- `solve`: Método que recebe um estado inicial do quebra-cabeça e resolve o problema utilizando o algoritmo de busca configurado. E pode receber os seguintes parâmetros opcionais:
  - `max_depth`: Um inteiro que limita a profundidade máxima da busca, evitando que o algoritmo explore estados muito profundos e potencialmente desnecessários.
  - `shuffle_actions`: Um booleano que indica se a ordem das ações geradas deve ser embaralhada aleatoriamente.
  - `random_state`: Um inteiro que define a semente aleatória para embaralhamento das ações, garantindo que o embaralhamento seja reproduzível.

  O método retorna um objeto `Result` contendo o caminho encontrado, o custo total, o estado inicial e o estado final se uma solução for encontrada, ou `None` se não houver solução.


#### Iterative solver

Herda de Solver e funciona exatamente igual, mas com a diferença de que o método `solve` executa a busca iterativamente incrementando a profundidade máxima da busca até encontrar uma solução ou atingir o limite de profundidade máximo. Isso é útil para algoritmos que não garantem encontrar a solução na primeira iteração, como a busca em profundidade iterativa (IDS).

#### Animação

Fizemos uma função para animar a solução do quebra-cabeça 8-puzzle, utilizando a biblioteca `plotly`. A função `animate_8puzzle_solution` recebe um objeto `Result` e gera uma animação interativa que mostra o progresso do agente na resolução do quebra-cabeça do estado inicial até o estado final. A animação exibe cada movimento do espaço vazio e as peças do quebra-cabeça, permitindo que o usuário visualize a solução passo a passo.

## Modelagem da Experimentação

Nesta seção vamos definir a  modelagem da implementação dos experimentos através da classe `Experiment`, das funções de custo `c1`, `c2`, `c3` e `c4`, das heurísticas `h1`, `h2`, `h3` e `h4`, das funções de pós-processamento dos experimentos para calcular o custo `calculate_costs` e `explode_costs` e dos algoritmos de busca que serão utilizados nos experimentos.

### Funções de Custo

Definimos quatro funções de custo diferentes para serem utilizadas nos experimentos, cada uma com uma lógica diferente para calcular o custo das ações realizadas pelo agente. Essas funções são usadas pelos algoritmos de busca para determinar o custo total da solução encontrada.

- `c1`: Todas as ações têm um custo de 2.
- `c2`: Ações verticais têm um custo de 2, ações horizontais têm um custo de 3.
- `c3`: Ações verticais têm um custo de 3, ações horizontais têm um custo de 2.
- `c4`: Todas as ações têm um custo de 2, exceto quando a posição do espaço vazio é (1, 1), onde o custo é 5.

### Heurísticas

Definimos duas heurísticas diferentes para serem utilizadas nos experimentos, baseados na descrição do trabalho, cada uma com uma lógica diferente para calcular o custo estimado de alcançar o estado objetivo a partir de um estado atual. Essas heurísticas são usadas pelos algoritmos de busca gulosa e A* para guiar a busca em direção à solução.

- `h1`: Retorna o número de peças fora de lugar vezes 2, ou seja, quantas peças estão na posição errada em relação aos estados objetivo.
- `h2`: Retorna a distância de Manhattan vezes 2, que é a soma das distâncias horizontais e verticais de cada peça em relação à sua posição no estado objetivo.

### Experiment

----

#### Instanciação

Modelamos a classe `Experiment` para representar um experimento de busca no quebra-cabeça 8-puzzle. A classe é parametrizada com os seguintes parâmetros:
- `algorithms`: Uma `list[Algorithm] | "all"` de algoritmos de busca a serem executados no experimento. Para cada algoritmo uma instância da classe `Solver`  será configurada com os parâmetros apropriados, no caso de "all", todos os algoritmos suportados serão utilizados. Os algoritmos suportados são: `bfs` (Busca em Largura), `dfs` (Busca em Profundidade), `dijkstra` (Busca de Custo Uniforme), `greedy` (Busca Gulosa) e `a_star` (Busca A*).

- `cfs`: Uma `list[CostFunction] | "all"` de funções de custo a serem utilizadas nos experimentos. Cada função de custo será usada pelos algoritmos de busca para calcular o custo total da solução encontrada. Se for "all", todas as funções de custo suportadas serão utilizadas. As funções de custo suportadas são: `c1`, `c2`, `c3` e `c4`.

- `hfs`: Uma `list[HeuristicFunction] | "all"` de funções heurísticas a serem utilizadas nos experimentos. Cada função heurística será usada pelos algoritmos. No caso de "all", todas as funções heurísticas suportadas serão utilizadas. As funções heurísticas suportadas são: `h1` e `h2`.

- `shuffle_actions`: Um booleano que indica se a ordem das ações geradas deve ser embaralhada aleatoriamente.

- `random_state`: `int | list[int] | None` que define a semente aleatória para embaralhamento das ações, garantindo que o embaralhamento seja reproduzível. Se for uma lista, cada experimento será executado com uma semente diferente, permitindo comparar os resultados de diferentes sementes aleatórias.

- `max_depth`: Um inteiro que limita a profundidade máxima da busca.

Então, ao instanciar a classe `Experiment`, serão criadas internamente as instâncias de `Solver` para cada combinação de algoritmo, função de custo e função heurística e, se aplicável, a semente aleatória para embaralhamento das ações. Isso permite que o agente execute os experimentos de forma flexível e extensível, testando diferentes combinações de algoritmos, funções de custo e heurísticas.

Logo, no exemplo de um experimento de **dijkstra** versus a **BFS**, com as funções de custo `c1` e `c2`, e os `random_state` 0 e 1, o experimento será configurado configurado com as seguintes combinações:
- `dijkstra` com `c1` e `random_state=0`
- `dijkstra` com `c1` e `random_state=1`
- `dijkstra` com `c2` e `random_state=0`
- `dijkstra` com `c2` e `random_state=1`
- `bfs` com `c1` e `random_state=0`
- `bfs` com `c1` e `random_state=1`
- `bfs` com `c2` e `random_state=0`
- `bfs` com `c2` e `random_state=1`

Tendo no final 8 instâncias de `Solver` configuradas para serem executadas no experimento.

Portanto, o número de combinações para cada combinação possível dos parâmetro será criada uma instância de `Solver` e o número total de instâncias criadas será o produto do número de algoritmos, funções de custo, funções heurísticas e sementes aleatórias.

---

#### Métodos

- `run`: Recebe uma lista de estados iniciais do quebra-cabeça e executa os algoritmos de busca configurados para cada estado inicial de forma Paralela usando `multiprocessing`. Retorna um `DataFrame` contendo os resultados dos experimentos, incluindo o tempo de execução, número de nós gerados, número de nós visitados, custo total e profundidade da solução para cada combinação de algoritmo, função de custo e função heurística.

    Para o caso de rodarmos o experimento para 30 estados iniciais para o experimento definido no exemplo acima, o `DataFrame` terá 240 linhas, onde cada linha representa uma combinação de algoritmo, função de custo, função heurística e estado inicial. As colunas do `DataFrame` serão:

    - **algorithm**: o nome do algoritmo de busca utilizado (por exemplo, `bfs`, `dfs`, `dijkstra`, `greedy`, `a_star`).
    - **cost_function**: o nome da função de custo utilizada (por exemplo, `c1`, `c2`, `c3`, `c4`).
    - **heuristic**: o nome da função heurística utilizada (por exemplo, `h1`, `h2`).
    - **raw**: o resultado bruto do experimento, que é um objeto `Result`.
    - **cost**: o custo total da solução encontrada, calculado pela função de custo utilizada.
    - **depth**: a profundidade da solução encontrada, que é o número de ações realizadas para alcançar o estado objetivo.
    - **max_depth**: o limite de profundidade máxima da busca, que é o parâmetro `max_depth` passado para o método `run`.
    - **visited**: o número de nós visitados durante a busca, ou seja, o número de estados explorados pelo algoritmo.
    - **generated**: o número de nós gerados durante a busca, ou seja, o número total de estados criados pelo algoritmo, incluindo os estados já visitados.
    - **time**: o tempo de execução do algoritmo em segundos, que é o tempo gasto para resolver o quebra-cabeça a partir do estado inicial até o estado objetivo.
    - **initial**: o estado inicial do quebra-cabeça
    - **final**: o estado final do quebra-cabeça, que é o estado objetivo alcançado pelo algoritmo.
    - **path**: uma lista de ações realizadas pelo agente para alcançar o estado objetivo a partir do estado inicial, ou seja, o caminho percorrido pelo agente para resolver o quebra-cabeça.
    - **random_state**: o valor da semente aleatória utilizada para embaralhar as ações, se aplicável. Se não for utilizado embaralhamento, o valor será `None`.


### Funções de Pós-processamento

- `calculate_costs`: Recebe um `DataFrame`  e uma função de custo e calcula o custo total para cada linha do `DataFrame` com base na função de custo fornecida. A função de custo deve ser uma das funções de custo definidas anteriormente (`c1`, `c2`, `c3` ou `c4`). O resultado é uma series com o custo total calculado para cada linha do `DataFrame`.

- `explode_costs`: Recebe um `DataFrame` e uma lista de funções de custo e expande o `DataFrame` e duplica o dataframe para cada função de custo, adicionando uma nova coluna `cost_function` com o nome da função de custo correspondente. O resultado é um novo `DataFrame` com as mesmas colunas do original, mas com uma linha para cada função de custo aplicada a cada linha do `DataFrame` original.

## Experimentos

Nesta seção, vamos executar os experimentos definidos na modelagem da experimentação, utilizando a classe `Experiment` e as funções de custo e heurísticas definidas anteriormente. Os experimentos serão executados para diferentes combinações de algoritmos, funções de custo e funções heurísticas, e os resultados serão armazenados em um `DataFrame` para análise posterior.

Os experimentos executados foram os definidos na atividade, que são:
- **Experimento 1**: BFS vs DFS vs Dijkstra para todas as funções de custo para 30 estados iniciais diferentes.
- **Experimento 2**: Dijkstra vs A* para todas as funções de custo e heurísticas para 30 estados iniciais diferentes.
- **Experimento 3**: BFS vs Dijkstra para todas as funções de custo e heurísticas para 30 estados iniciais diferentes.
- **Experimento 4**: BFS vs DFS aleatoriamente embaralhando as ações para 15 estados iniciais diferentes e 10 sementes aleatórias diferentes.

---

### Estados

Para os experimentos, utilizamos 30 estados iniciais aleatórios gerados a partir de uma semente aleatória, garantindo que os estados sejam reproduzíveis. Esses estados foram gerados utilizando a função `Problem.make_solvable` com uma semente aleatória fixa, garantindo que os estados sejam solucionáveis e representem uma variedade de configurações do quebra-cabeça 8-puzzle.

### Experimento 1

O primeiro experimento compara os algoritmos de busca BFS, DFS e Dijkstra para todas as funções de custo definidas (`c1`, `c2`, `c3` e `c4`) em 30 estados iniciais diferentes. O objetivo é analisar o desempenho de cada algoritmo em termos de tempo de execução, número de nós gerados, número de nós visitados, custo total e profundidade da solução.

Logo, teremos 12 combinações de algoritmos, funções de custo e estados iniciais, resultando em um total de 360 experimentos (30 estados iniciais x 12 combinações).

### Experimento 2

Já o segundo experimento compara os algoritmos de busca Dijkstra e A*. No caso de A*, utilizamos as heurísticas `h1` e `h2` e todas as funções de custos definidas anteriormente. Para Dijkstra, utilizamos todas as funções de custo definidas anteriormente. O objetivo é analisar o desempenho de cada algoritmo em termos de tempo de execução, número de nós gerados, número de nós visitados, custo total e profundidade da solução.

No final, teremos  4 combinações de Dijkstra e 8 combinações de A* (com as duas heurísticas), resultando em um total de 240 experimentos (30 estados iniciais x 12 combinações).

### Experimento 3

O terceiro experimento compara os algoritmos de busca A* e a Busca Gulosa (Greedy). No caso de A*, utilizamos as heurísticas `h1` e `h2` e todas as funções de custos definidas anteriormente. Para a Busca Gulosa, utilizamos todas as heurísticas definidas anteriormente. O objetivo é analisar o desempenho de cada algoritmo em termos de tempo de execução, número de nós gerados, número de nós visitados, custo total e profundidade da solução.

No final, teremos 2 combinações da Busca Gulosa e 8 combinações de A* (com as duas heurísticas), resultando em um total de 300 experimentos (30 estados iniciais x 10 combinações).

Depois de executar o algoritmo guloso, pós-processamos os resultado para calcular o custo para cada execução para cada função de custo definida anteriormente então o número final de registros será 240 do A* + 240 do Guloso Expandido = 480 experimentos.

### Experimento 4

O último experimento compara os algoritmos de busca BFS e DFS, utilizando ações embaralhadas aleatoriamente. O objetivo é analisar o desempenho de cada algoritmo em termos de tempo de execução, número de nós gerados, número de nós visitados, custo total e profundidade da solução.

Para este experimento, utilizamos 15 estados iniciais diferentes e 10 sementes aleatórias diferentes para embaralhar as ações. O embaralhamento das ações é feito utilizando no método `Solver.solve` com uma semente aleatória fixa, garantindo que os estados sejam reproduzíveis. O objetivo é analisar o desempenho de cada algoritmo em termos de tempo de execução, número de nós gerados, número de nós visitados, custo total e profundidade da solução.

No final, teremos 2 combinações de BFS e DFS (com ações embaralhadas) e 15 estados iniciais diferentes, resultando em um total de 300 experimentos (15 estados iniciais x 20 combinações).

Depois de executar, pós-processamos e calculamos os custos de cada execução para cada função de custo definida, resultando em um total de 1200 experimentos (300 experimentos x 4 funções de custo).

### Salvando os resultados

Por fim, salvamos os resultados dos experimentos em arquivos .parquet para cada experimento, permitindo que os resultados sejam facilmente carregados e analisados posteriormente.

Utilizamos pickle para salvar  a coluna `raw` do DataFrame, que contém os objetos `Result` de cada experimento. Isso permite que os resultados sejam carregados posteriormente sem a necessidade de recalcular os experimentos, economizando tempo e recursos computacionais.

Escolhemos o formato Parquet para salvar os resultados dos experimentos devido à sua eficiência em termos de armazenamento e leitura, além de ser um formato amplamente utilizado para análise de dados em larga escala. O Parquet é um formato colunar, o que permite uma leitura mais rápida e eficiente dos dados, especialmente quando se trabalha com grandes volumes de dados.
