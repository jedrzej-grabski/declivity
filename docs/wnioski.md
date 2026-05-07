### 1 - Błędy w implementacji
Przy okazji doczytywania w badaniu, zauwazyłem ze wkradł się błąd implementacyjny.
Mocno skupiliśmy się na uzywaniu macierzy jednostkowej w liczeniu wektora spadku, a okazuję się ze nie jest
to jedyne miejsce w którym występuje theta. W minimalizacji podprzestrzeni nadal uzywaliśmy (niewprost) macierzy jednostkowej. Dodałem tam korzystanie z naszej macierzy hesjanu, to dało zauważalne przyspieszenie.

Teraz przy podaniu początkowej macierzy wykorzystujemy ją w każdym fragmencie algorytmu.

### 2 - Dlaczego tylko diagonala?
Zastanawialiśmy się dlaczego korzystamy przy zastąpieniu macierzy jednostkowej korzystaliśmy jedynie z diagonali, a nie z całej macierzy. Okazuje się że to jednak ma swoje konsekwencje.

#### 2.1
Gdy liczymy punkt Cauchy'ego, normalnie liczymy pochodną B po d: theta * d'B_0 d - d' W M W' d, dla pełnej macierzy wykonujemy iloczyn macierz wektor co daje nam n^2 .

#### 2.2
Gdy minimalizujemy podprzestrzeń, we wzorze Woodbury'ego, też dzielimy wektory przez tą naszą diagonalę (lub macierz) przeskalowaną przez theta

#### 2.3
Przy aktualizacji par korekcyjnych, też iloczyny macierz-wektor w przypadku pełnej macierzy

Zastosowanie całej macierzy wydaję się na wielu etapach mocno zwiększac koszt obliczeń.

### 3 - Rotacje funckji testowych
W celu sprawdzenia kiedy zastosowanie całej macierzy może byc korzystne (lub kiedy bazowanie na samej diagonali może być szkodliwe) przygotowałem takie funckję testowe:

Wyliczyłem takie macierze rotacji:

    === uniform_45 ===
     [[ 0.7071 -0.7071  0.      0.      0.      0.      0.      0.      0.      0.    ]
      [ 0.5     0.5    -0.7071  0.      0.      0.      0.      0.      0.      0.    ]
      [ 0.3536  0.3536  0.5    -0.7071  0.      0.      0.      0.      0.      0.    ]
      [ 0.25    0.25    0.3536  0.5    -0.7071  0.      0.      0.      0.      0.    ]
      [ 0.1768  0.1768  0.25    0.3536  0.5    -0.7071  0.      0.      0.      0.    ]
      [ 0.125   0.125   0.1768  0.25    0.3536  0.5    -0.7071  0.      0.      0.    ]
      [ 0.0884  0.0884  0.125   0.1768  0.25    0.3536  0.5    -0.7071  0.      0.    ]
      [ 0.0625  0.0625  0.0884  0.125   0.1768  0.25    0.3536  0.5    -0.7071  0.    ]
      [ 0.0442  0.0442  0.0625  0.0884  0.125   0.1768  0.25    0.3536  0.5    -0.7071]
      [ 0.0442  0.0442  0.0625  0.0884  0.125   0.1768  0.25    0.3536  0.5     0.7071]]
     Orthogonal check (R R' = I): max error = 2.78e-17
     det(R) = 1.0000

     === golden ===
     [[-0.7373 -0.6756  0.      0.      0.      0.      0.      0.      0.      0.    ]
      [ 0.0589 -0.0643  0.9962  0.      0.      0.      0.      0.      0.      0.    ]
      [-0.4097  0.4471  0.0531 -0.7934  0.      0.      0.      0.      0.      0.    ]
      [ 0.5258 -0.5738 -0.0681 -0.5995  0.1736  0.      0.      0.      0.      0.    ]
      [ 0.0782 -0.0853 -0.0101 -0.0892 -0.8306  0.5373  0.      0.      0.      0.    ]
      [ 0.0129 -0.0141 -0.0017 -0.0147 -0.137  -0.2183 -0.9659  0.      0.      0.    ]
      [ 0.0222 -0.0242 -0.0029 -0.0253 -0.236  -0.3762  0.1195  0.887   0.      0.    ]
      [ 0.0401 -0.0438 -0.0052 -0.0457 -0.426  -0.679   0.2157 -0.4339 -0.342   0.    ]
      [-0.0135  0.0147  0.0017  0.0154  0.1433  0.2283 -0.0725  0.1459 -0.8682 -0.3827]
      [ 0.0056 -0.0061 -0.0007 -0.0064 -0.0593 -0.0946  0.03   -0.0604  0.3596 -0.9239]]
     Orthogonal check (R R' = I): max error = 2.22e-16
     det(R) = 1.0000

     === random ===
     [[ 0.0909 -0.5112  0.2211  0.5329 -0.3343  0.0353  0.1921  0.203  -0.4256  0.1593]
      [ 0.2624  0.2968  0.1433  0.6909  0.2092 -0.1257 -0.1786  0.2145  0.4337 -0.1398]
      [-0.0552 -0.305   0.4387 -0.1875 -0.043  -0.2788  0.3856 -0.0538  0.6089  0.2727]
      [ 0.639  -0.3559 -0.1744 -0.1737  0.528  -0.102  -0.1407  0.0668 -0.0913  0.2874]
      [ 0.2218  0.1973 -0.1956  0.2461  0.0319  0.4462  0.4721 -0.5601  0.0745  0.2661]
      [ 0.0863  0.2737 -0.5389 -0.0458 -0.3206 -0.2645  0.2951  0.4916  0.0783  0.3427]
      [-0.5021 -0.0271 -0.0322  0.1513  0.2074  0.2445 -0.3795  0.0897  0.0812  0.6796]
      [-0.3806 -0.4329 -0.5565  0.267   0.2595 -0.2641  0.1529 -0.1712  0.1415 -0.2774]
      [ 0.1363 -0.3456 -0.2042 -0.1109 -0.2587  0.6343 -0.1465  0.2769  0.4393 -0.2181]
      [ 0.1985 -0.0976 -0.163   0.0652 -0.5317 -0.3063 -0.5137 -0.4858  0.1425  0.1532]]
     Orthogonal check (R R' = I): max error = 8.88e-16
     det(R) = 1.0000


Odpowiadają one odpowiednio trzem strategiom rotacji:

Uniform 45° -
łańcuch rotacji w kolejnych płaszczyznach (0,1), (1,2)... każda o kąt 45°

Golden angle -
łancuch rotacji z różnymi kątami (k+1) * 137.5° dla płaszczyzny k.
Miało to na celu wykluczuć powtarzalność w rotacjach, z uwagi na brak zwielokrotniania się 
się kątów w wielokrotnościach.

Random -
Wygenerowałem macierz nxn z rozkładem normalnym, poddałem rozkładowi QR i zastosowałem czynnik Q 
jako naszą macierz obrotu. Wydaję mi się że dzięki temu powinniśmy zostać prawidziwe losową zrotowaną funckję przy zachowaniu ortogonalności macierzy i wyznacznika 1.

Dla hesjanów tych obróconych funkcji możemy policzyć frakcję energi diagonalnej na poziomie:

| typ rotacji | Frakcja diag. (10D) | Frakcja diag. (50D) |
|---|---:|---:|
| Brak | 1.00 | 1.00 |
| Uniform 45° | 0.49 | 0.91 |
| Golden angle | 0.83 | 0.94 |
| Random | 0.26 |  0.18 |


### 4 - Eksperymenty z pełnym hesjanem

#### 4.1 - Zmiany w implementacji

Aby umożliwić ekserymenty z hesjanem dodałem abstrakcję nad początkowym paremetrem pełniącym rolę przekątnej hesjanu, hesjanu, lub macierzy jednostkowej. Teraz tworzymy obiekt początkowy w odpowiedniu sposób, ale algorytm w każdym wariancie działa tak samo.

#### 4.2 - Wyniki

Uruchomiłem nowe poobracane funckje w dwóch rozmiarach, dla trzech początkowych rodzajów hesjanu:

- **n = 10, m = 10**: gdzie poprawki mogą opisać nam 2m = 20 kierunków, a więc pokrywa nasze 10D co sprawia że początkowy hesjan jest mało ważny, ponieważ poprawki w pełni uczą się kształtu funkcji
- **n = 50, m = 5**: Tutaj zakładałem, że przy znacząco mniejszym rozmiarze okna historii od wymiaru funkcji, dobra aproksymacja hesjanu może mieć zdecydowanie większe znaczenie.


##### n = 50, m = 5

![50D summary](../plots/lbfgsb/rotation_study/50D_m5_summary.png)

| Obrót | M. Jednostkowa | Przekątna | Pełny Hesjan |
|---|---|---|---|
| None (axis-aligned) | 5,064 | **23** | 23 |
| Uniform 45° | 3,508 | 10,000+ | **22** |
| Golden angle | 5,968 | 10,000+ | **22** |
| Random | 6,111 | 10,000+ | **23** |

Widzimy że pełny Hesjan sprawia że wchodzimy w margines minimum w bardzo szybkim czasie, niezależnie od rotacji.

Diagonala oczywiście, zawierając de facto błędne informację, robi nam zdecydowanie pod górkę.

Macierz jednostkowa reprezentuje klasyczne zachowanie uczenia się kształtu funkcji od zera.

##### n = 10, m = 10

![10D summary](../plots/lbfgsb/rotation_study/10D_m10_summary.png)

Gdy poprawki obejmują całą przestrzeń pełny hesjan nadal daje przewagę, ale nie aż tak rażącą.

##### Zbieżności 50D, obrót *Random*

![Random 50D convergence](../plots/lbfgsb/rotation_study/50D_m5_random_convergence.png)

### 5 - Eksperymenty CMA-ES -> L-BFGS

Głowiliśmy się, dlaczego jednak podajemy odwortność macierzy kowariancji z CMAES-a. To też powtierdził się w eksperymentach poniżej. Otóż problem tkwił w różnicy między L-BFGS i L-BFGS-B: W wersji bez ograniczeń faktycznie w praktyce trzymamy odwrotność hesjanu, jednak w wersji ograniczonej, przetrzymujemy aproksymację faktycznego hesjanu i używamy jej do liczenia punkty Cauchyego, a odwracamy go "na życzenie".


Zgodnie z planem testowałem 6 transformacji: macierz jednostkowa, bezpośrednio C, odwrotność C^{-1}, odwrotność przeskalowana (sigma^2 C)^{-1}, znormalizowana C/tr(C)*n oraz odwrócona znormalizowana (C/tr(C)*n)^{-1}.

#### 50D, m=5, 300 generacji CMA-ES

![Uniform 45 convergence 50D](../plots/hybrid/handoff_study/uniform_45_50d_convergence.png)

![Random convergence 50D](../plots/hybrid/handoff_study/random_50d_convergence.png)

#### 10D, m=5, 100 generacji CMA-ES

![Uniform 45 convergence 10D](../plots/hybrid/handoff_study/uniform_45_10d_convergence.png)

![Random convergence 10D](../plots/hybrid/handoff_study/random_10d_convergence.png)

Jak widać faktycznie potrzebujemy odwróconej macierzy żeby osiągnąć poprawę.

Skalowanie prawie nie ma znaczenie, theta i tak dostosowuje skalę globalną w kilku iteracjach