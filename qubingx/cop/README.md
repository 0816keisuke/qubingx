# Formulation

## Quadratic Assignment Problem (QAP)

- $F$: the number of cities
- $w(f_i, f_j)$: the weight between facility $f_i$ and $f_j$
- $d(l_k, l_l)$: the distance between location $l_k$ and $l_l$
- $x_{i,k}$: bellow

$$
\begin{align}
    x_{i,k} =
    \begin{cases}
        0 & \text{if facility } f_i \text{ is assigned to location } l_k \\
        1 & \text{Otherwise}
    \end{cases}
\end{align}
$$
The energy function of TSP is as follows:
$$
\begin{align}
    \mathcal{H}_{\mathrm{all}} &= \mathcal{H}_{\mathrm{obj}} + \alpha \left( \mathcal{H}_{\mathrm{pen1}} + \mathcal{H}_{\mathrm{pen2}} \right) \\

    \mathcal{H}_{\mathrm{obj}} &= \sum_{i=1}^F \sum_{j=1}^F \sum_{j=1}^F \sum_{l=1}^F w(f_i,f_j) d(l_k, l_l) x_{i,k} x_{j,l} \\

    \mathcal{H}_{\mathrm{pen1}} &= \sum_{i=1}^F \left( \sum_{k=1}^F x_{i,k} - 1 \right)^2 \\
    &= \sum_{i=1}^F \left( 2 \sum_{k=1}^{F-1} \sum_{l=k+1}^F x_{i,k} x_{i,l} - \sum_{k=1}^F x_{i,k} + 1 \right) \\
    &= 2 \sum_{i=1}^F \sum_{k=1}^{F-1} \sum_{l=k+1}^F x_{i,k} x_{i,l} - \sum_{i=1}^F \sum_{k=1}^F x_{i,k} + F \\

    \mathcal{H}_{\mathrm{pen2}} &= \sum_{k=1}^F \left( \sum_{i=1}^F x_{i,k} - 1 \right)^2 \\
    &= \sum_{k=1}^F \left( 2 \sum_{i=1}^{F-1} \sum_{j=i+1}^F x_{i,k} x_{j,k} - \sum_{k=1}^F x_{i,k} + 1 \right) \\
    &= 2 \sum_{k=1}^F \sum_{i=1}^{F-1} \sum_{j=i+1}^F x_{i,k} x_{j,k} - \sum_{k=1}^F \sum_{i=1}^F x_{i,k} + F
\end{align}
$$

## Travelling Salesman Problem (TSP)

- $N$: the number of cities
- $d(u, v)$: the distance between city $u$ and $v$
- $x_{t,u}$: below

$$
\begin{align}
    x_{t,u} =
    \begin{cases}
        0 & \text{if visit city } u \text{ at time } t \\
        1 & \text{Otherwise}
    \end{cases}
\end{align}
$$
The energy function of TSP is as follows:
$$
\begin{align}
    \mathcal{H}_{\mathrm{all}} &= \mathcal{H}_{\mathrm{obj}} + \alpha \left( \mathcal{H}_{\mathrm{pen1}} + \mathcal{H}_{\mathrm{pen2}} \right) \\

    \mathcal{H}_{\mathrm{obj}} &= \sum_{t=1}^N \sum_{u=1}^N \sum_{v=1}^N d(u,v) x_{t,u} x_{t+1,v} \\

    \mathcal{H}_{\mathrm{pen1}} &= \sum_{t=1}^N \left( \sum_{u=1}^N x_{t,u} - 1 \right)^2 \\
    &= \sum_{t=1}^N \left( 2 \sum_{u=1}^{N-1} \sum_{v=u+1}^N x_{t,u} x_{t,v} - \sum_{u=1}^N x_{t,u} + 1 \right) \\
    &= 2 \sum_{t=1}^N \sum_{u=1}^{N-1} \sum_{v=u+1}^N x_{t,u} x_{t,v} - \sum_{t=1}^N \sum_{u=1}^N x_{t,u} + N \\

    \mathcal{H}_{\mathrm{pen2}} &= \sum_{u=1}^N \left( \sum_{t=1}^N x_{t,u} - 1 \right)^2 \\
    &= \sum_{u=1}^N \left( 2 \sum_{t=1}^{N-1} \sum_{t'=t+1}^N x_{t,u} x_{t',u} - \sum_{t=1}^N x_{t,u} + 1 \right) \\
    &= 2 \sum_{u=1}^N \sum_{t=1}^{N-1} \sum_{t'=t+1}^N x_{t,u} x_{t',u} - \sum_{u=1}^N \sum_{t=1}^N x_{t,u} + N
\end{align}
$$

## Slot-placement Problem (SPP)

- $m$: the number of items
- $t$: the number of slots
- $w(c_i, c_j)$: the number of wiring between item $c_i$ and $c_j$
- $d(s_a, s_b)$: the manhattan-distance between slot $s_a$ and $s_b$
- $x_{i,a}$: below

$$
\begin{align}
    x_{i,a} =
    \begin{cases}
        0 & \text{if } c_i \text{ is assigned to the slot } s_a \\
        1 & \text{Otherwise}
    \end{cases}
\end{align}
$$
The energy function of SPP is as follows:
$$
\begin{align}
    \mathcal{H}_{\mathrm{all}} &= \mathcal{H}_{\mathrm{obj}} + \alpha \mathcal{H}_{\mathrm{pen1}} + \beta \mathcal{H}_{\mathrm{pen2}} \\

    \mathcal{H}_{\mathrm{obj}} &= \sum_{a=1}^t \sum_{i=1}^m \sum_{b=1}^t \sum_{j=1}^N w(c_i, c_j) d(s_a,s_b) x_{i,a} x_{j,b} \\

    \mathcal{H}_{\mathrm{pen1}} &= \sum_{i=1}^m \left( \sum_{a=1}^t x_{i,a} - 1 \right)^2 \\
    &= \sum_{i=1}^m \left( 2 \sum_{a=1}^{t-1} \sum_{b=a+1}^t x_{i,a} x_{i,b} - \sum_{a=1}^t x_{i,a} + 1 \right) \\
    &= 2 \sum_{i=1}^m \sum_{a=1}^{t-1} \sum_{b=a+1}^t x_{i,a} x_{i,b} - \sum_{i=1}^m \sum_{a=1}^t x_{i,a} + m \\

    \mathcal{H}_{\mathrm{pen2}} &= \sum_{a=1}^t \left( \sum_{i=1}^m x_{i,a} - \frac{1}{2} \right)^2 - \frac{t}{4} \\
    &= \sum_{a=1}^t \left( 2 \sum_{i=1}^{m-1} \sum_{j=i+1}^m x_{i,a} x_{j,a} + \frac{t}{4} \right) - \frac{t}{4} \\
    &= 2 \sum_{a=1}^t \sum_{i=1}^{m-1} \sum_{j=i+1}^m x_{i,a} x_{j,a}
\end{align}
$$

## Knapsack problem

- $N$: number of items
- $v_i$: the value of item $i$
- $w_i$: the weight of item $i$

### 1-hot encoding

### Binary encoding

- $D$: $\log_2 \lfloor W-1 \rfloor$

$$
\begin{align}
    \mathcal{H}_{\mathrm{all}} &= \mathcal{H}_{\mathrm{obj}} + \alpha \mathcal{H}_{\mathrm{pen}} \\

    \mathcal{H}_{\mathrm{all}} &= - \sum_{i=1}^N v_ix_i \\

    \mathcal{H}_{\mathrm{pen}} &= \left( W - \sum_{i=1}^N w_i x_i - \sum_{j=1}^{D} 2^j y_j \right)^2 \\
    &= W^2 - 2 W \sum_{i=1}^{N} w_i x_i - 2 W \sum_{j=1}^D 2^j y_j \\
    & ~~~~~~~~~~ + \left( \sum_{i=1}^{N} w_i x_i \right)^2 + 2 \sum_{i=1}^{N} w_i x_i \sum_{j=1}^D 2^j y_j + \left( \sum_{j=1}^D 2^j y_j \right)^2 \\
    &= W^2 - 2 W \sum_{i=1}^{N} w_i x_i - 2 W \sum_{j=1}^D 2^j y_j \\
    & ~~~~~~~~~~ + \sum_{i=1}^{N-1} \sum_{j=i+1}^{N-1} 2 w_i w_j x_i x_j + \sum_{a=1}^{N} w_i^2 x_i \\
    & ~~~~~~~~~~ + \sum_{i=1}^N \sum_{j=1}^D w_i 2^{j+1} x_i y_j + \sum_{j=1}^D \sum_{k=j+1}^D 2^{j+1} 2^k y_j y_k + \sum_{j=1}^D \left(2^{2j} \right) y_j
\end{align}
$$

### Unary encoding
