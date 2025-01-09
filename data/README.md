# Definition of multi-behavior recommendation data and scenarios

## Data of multi-behavior recommendation systems

For a multi-behavior recommendation dataset $D$, the dataset generally has user set $U$, item set $I$, user-item interaction set $R$ and behavior type set $\mathcal{B}=\\{b\_1,b\_2,\dots\\}$.

For an interaction $R\_{ui}$ between a specific user $u$ and item $i$, the interaction has multiple behaviors record, denoted as $B\_{ui}=\\{{b\_{ui}}\_1, {b\_{ui}}\_2, \dots \\}$, where every behavior ${b\_{ui}}\_j \in \mathcal{B}$.

Based on the common recommendation system model training and evaluation process, multi-behavior data can be divided according to interaction $R$. Given a split ratio $r(0<r<1)$, the dataset is split into $D_{\text{Train}}$ and $D_{\text{Test}}$, where $|R_{\text{Train}}|/|R| = r$.

## Model training for multi-behavior recommendation systems

For a multi-behavior recommendation model $M_{\theta}$, the model is able to recommend a set of $N$ items to a user $u$:

$$
I_{u,N}=M_{\theta}(u, I, N),
$$

where $I_{u,N}$ is the recommendation result and  $u \in U$.

The model parameters $\theta$ of the recommendation model $M_{\theta}$ are updated by the multi-behavior training set $D_{\text{Train}}$ and the designed loss function $\mathcal{L}(\theta)$.

## Model evaluation of multi-behavior recommendation systems

For a trained recommendation model $M_{\theta^\*}$, the model is used to recommend $N$ item sets $I_{u,N}^\*=M_{\theta^\*}(u,I,N)$ to all users $u$, and the recommendation results are evaluated. Since we have multiple type of behaviors, we can evaluate the recommendation results from multiple behavioral perspectives:

$$
E_{u, b_i}@N=f(I_{u,N}^*, D_{\text{Test}, u}, b_i, N),
$$

where $f$ is an evaluation function, $D_{\text{Test}, u}$ is the real record of user $u$'s interaction, $u \in U$ and $b_i \in \mathcal{B}$.

So the recommendation performance of the recommender system in this behavior $b_i$:

$$
E_{b_i}@N=\frac{1}{|U|} \sum_{u\in U} E_{u, b_i}@N.
$$