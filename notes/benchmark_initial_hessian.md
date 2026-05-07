# Benchmark: Wpływ początkowej aproksymacji Hesjanu na zbieżność L-BFGS-B

## Opis eksperymentu

Funkcja testowa: **Ellipsoid (10D)**, liczba uwarunkowania = 10^6.
Diagonala Hesjanu: od 2 (zmienna 1) do 2 000 000 (zmienna 10).

Porównano trzy warianty początkowej aproksymacji Hesjanu `B_0`:

| Wariant | `initial_hessian` | Opis |
|---------|-------------------|------|
| Macierz jednostkowa (domyślnie) | `None` | `B_0 = I` --- brak wiedzy o krzywiźnie |
| Dokładna diagonala | `2 * 10^(6i/9)` | `B_0 = diag(h)` --- dokładna krzywizna per zmienna |
| Skalar (średnia krzywizna) | `~222 444` | `B_0 = c * I` --- przybliżona skala globalna |

Wszystkie warianty z gradientem analitycznym, `pgtol = 1e-12`, `factr = 0`.

## Wyniki

| Wariant | Ewaluacje | Iteracje | Końcowe f(x) |
|---------|-----------|----------|---------------|
| Macierz jednostkowa | **14 109** | 7 570 | 1.39e-28 |
| Dokładna diagonala | **881** | 779 | 2.67e-26 |
| Skalar (średnia) | **913** | 808 | 1.82e-28 |

Dokładna diagonala jest **16x szybsza** niż domyślna macierz jednostkowa.

## Analiza wykresów

### Zbieżność po ewaluacjach (lewy górny panel)

Krzywe z informacją o Hesjanie (zielona, niebieska) spadają do 1e-25 w ~900
ewaluacjach. Krzywa z macierzą jednostkową (czerwona) potrzebuje wszystkich
14 109 ewaluacji. Początkowy stromy spadek jest podobny --- różnica pojawia
się w długim "plateau", gdzie wariant domyślny dreptał w miejscu.

### Zbieżność po iteracjach (prawy górny panel)

Per iteracja różnice są mniejsze niż per ewaluacja. Wariant domyślny robi
postęp w każdej iteracji, ale potrzebuje 7 500 iteracji. Warianty z informacją
o Hesjanie kończą w ~800. Wniosek: macierz jednostkowa nie powoduje gorszych
kroków per se, ale wymaga znacznie więcej kroków, ponieważ punkt Cauchy'ego
jest źle przeskalowany.

### Norma rzutowanego gradientu (lewy dolny panel)

Wariant domyślny (czerwony) wykazuje chaotyczne oscylacje normy gradientu
między 1e-2 a 1e-11 przez tysiące ewaluacji. Warianty z informacją o Hesjanie
spadają płynnie do 1e-11. Te oscylacje to sygnatura źle uwarunkowanego
punktu Cauchy'ego --- algorytm oscyluje między zmiennymi o drastycznie różnej
krzywiźnie.

### Długość kroku (prawy dolny panel)

**Najbardziej wymowny panel.** Warianty z informacją o Hesjanie szybko
stabilizują się przy długości kroku ~1 (ideał dla metod quasi-Newtonowskich).
Wariant domyślny jest chaotyczny --- długości kroków wahają się o 10 rzędów
wielkości przez cały przebieg. Oznacza to, że line search ciągle walczy,
bo kierunek z punktu Cauchy'ego jest źle przeskalowany.

## Dlaczego plateau?

Kluczowe pytanie: dlaczego wariant z macierzą jednostkową ma długie plateau
w zbieżności?

Odpowiedź leży w interakcji między `theta` a krzywiznami zmiennych:

**Faza 1 (stromy spadek):** Zmienne o dużej krzywiźnie (9, 10) mają ogromne
gradienty, więc algorytm atakuje je pierwsze. Wartość funkcji spada szybko,
bo te zmienne dominują `f(x)`.

**Faza 2 (plateau):** Zmienne o dużej krzywiźnie są już blisko zera. Pozostały
błąd tkwi w zmiennych o małej krzywiźnie (1, 2, 3). Problem: `theta = y'y / y's`
zostało skalibrowane na podstawie ostatnich kroków, które były zdominowane
przez zmienne o dużej krzywiźnie. Dlatego `theta ≈ 10^6`.

Efektywna baza Hesjanu to `theta * I = 10^6 * I`. Dla zmiennej 1 (rzeczywista
krzywizna = 2) algorytm myśli, że krzywizna wynosi 10^6. Punkt Cauchy'ego
wykonuje kroki **500 000 razy za małe** dla tej zmiennej.

Korekcje L-BFGS (część `W*M*W'`) próbują to kompensować, ale przy zaledwie
`m = 10` parach korekcyjnych i liczbie uwarunkowania 10^6 nie są w stanie
w pełni nauczyć się dysproporcji. Algorytm posuwał się milimetrowo po
zmiennych o małej krzywiźnie, tysiącami iteracji.

**Z dokładną diagonalą**, `h_diag = theta * [2, 9.3, ..., 2e6]`. Gdy theta
rośnie po krokach na zmiennych o dużej krzywiźnie, *względne* skalowanie
między zmiennymi jest zachowane. Zmienna 1 zawsze dostaje kroki 10^6 razy
większe niż zmienna 10, co jest poprawne. Brak plateau.

## Wniosek

Nawet przybliżona estymata skalarną średnią krzywizny pomaga ogromnie --- przesuwa
bazę `theta` bliżej właściwej skali, dzięki czemu rozbieżność między `theta`
a zmiennymi o małej krzywiźnie nie jest tak ekstremalna.

Parametr `initial_hessian` ma trwały wpływ na cały przebieg optymalizacji
(nie tylko na pierwszą iterację), ponieważ względne skalowanie per zmienna
jest mnożone przez adaptacyjne `theta` w każdym punkcie Cauchy'ego.

---

## Dodatkowe wyjaśnienia

### Co tak naprawdę robimy, przekazując macierz?

Metoda Newtona wyznacza kierunek kroku jako `d = -H^{-1} * g`, gdzie `H` to
hesjan (macierz drugich pochodnych), a `g` to gradient. Hesjan mówi algorytmowi:
"w tym kierunku funkcja zakrzywia się ostro (duże kroki są niebezpieczne),
a w tamtym kierunku jest płaska (potrzeba dużych kroków)".

L-BFGS-B nie zna prawdziwego hesjanu --- przybliża go za pomocą par korekcyjnych
`(s, y)` zebranych w trakcie optymalizacji. Ale na **pierwszej iteracji** nie ma
jeszcze żadnych par. Algorytm potrzebuje punktu startowego dla tej aproksymacji.
Domyślnie przyjmuje `B_0 = I` (macierz jednostkowa), co oznacza: "zakładam,
że krzywizna wynosi 1 we wszystkich kierunkach".

Przekazując `initial_hessian`, mówimy algorytmowi: "zanim czegokolwiek się
nauczysz z danych, oto moje wstępne przekonanie o krzywiźnie funkcji".
Na przykład:

- `initial_hessian=100.0` --- "krzywizna jest mniej więcej 100 wszędzie"
- `initial_hessian=np.array([2, 50, 1e6])` --- "zmienna 1 ma krzywiznę 2,
  zmienna 2 ma 50, zmienna 3 ma milion"

To **względne skalowanie** między zmiennymi jest kluczowe. Utrzymuje się przez
cały przebieg optymalizacji, ponieważ efektywna baza hesjanu to
`theta * diag(initial_hessian)`. Współczynnik `theta` adaptuje się (zmienia
skalę globalną), ale proporcje między zmiennymi z `initial_hessian` pozostają.

W praktyce: jeśli wiesz, że jedna zmienna jest "sztywna" (duża krzywizna),
a inna "miękka" (mała krzywizna), przekazanie tej informacji pozwala algorytmowi
od razu wykonywać kroki o właściwych rozmiarach w każdym kierunku, zamiast
odkrywać to metodą prób i błędów przez tysiące iteracji.

### Czym jest norma rzutowanego gradientu?

Zwykły gradient `g = nabla f(x)` mówi: "funkcja maleje najszybciej w tym
kierunku". Ale w optymalizacji z ograniczeniami (bound constraints) nie
możemy podążać za gradientem w dowolnym kierunku --- jeśli `x_i` jest już na
granicy `l_i` i gradient każe iść jeszcze niżej, to nie możemy.

**Rzutowany gradient** koryguje tę sytuację. Dla każdej zmiennej:

- Jeśli `x_i` leży na dolnej granicy `l_i` i gradient wskazuje w dół
  (`g_i > 0`, chce zmniejszyć `x_i`) --- rzutowany gradient jest zerowany:
  "nie mogę iść dalej w dół, jestem na ścianie"
- Jeśli `x_i` leży na górnej granicy `u_i` i gradient wskazuje w górę
  (`g_i < 0`, chce zwiększyć `x_i`) --- analogicznie zerowany
- W pozostałych przypadkach --- rzutowany gradient = zwykły gradient

Norma nieskończoności (`||proj g||_inf = max|pg_i|`) to **najgorszy komponent**
rzutowanego gradientu. Gdy wynosi zero, spełnione są warunki KKT
(Karush-Kuhn-Tucker) --- punkt jest optymalny z uwzględnieniem ograniczeń.

To jest główne **kryterium zbieżności** algorytmu L-BFGS-B: zatrzymaj się,
gdy `||proj g||_inf <= pgtol`.

### Dlaczego długość kroku jest ważna i dlaczego jest chaotyczna dla macierzy jednostkowej?

**Dlaczego zwracamy na nią uwagę:**

W metodach quasi-Newtonowskich (BFGS, L-BFGS) idealny krok ma długość
`alpha = 1`. Dlaczego? Bo aproksymacja hesjanu `B` jest budowana tak, aby
kwadratowy model `Q(d) = g'd + (1/2)d'Bd` dobrze przewidywał rzeczywistą
funkcję. Jeśli model jest dokładny, to minimum modelu (`d = -B^{-1}g`)
powinno leżeć blisko minimum funkcji wzdłuż tego kierunku. Innymi słowy:
`f(x + 1.0 * d) ≈ min_alpha f(x + alpha * d)`.

Gdy line search regularnie akceptuje `alpha ≈ 1`, to sygnał, że aproksymacja
hesjanu jest dobra --- model dobrze przewiduje funkcję.

Gdy `alpha` odbiega od 1 (jest dużo mniejsze lub większe), to sygnał, że
model jest niedokładny --- line search musi korygować złe przewidywania modelu.

**Dlaczego jest chaotyczna dla macierzy jednostkowej:**

Z `B_0 = I` na Ellipsoidzie, efektywny hesjan to `theta * I`. Ale `theta`
jest jednym skalarem dla wszystkich zmiennych. Rozważmy co się dzieje:

1. Algorytm robi krok zdominowany przez zmienną 10 (krzywizna 2e6).
   `theta` kalibruje się na ~10^6.

2. Następna iteracja: algorytm próbuje ruszyć zmienną 1 (krzywizna 2).
   Model myśli, że krzywizna wynosi 10^6, więc proponuje krok 500 000x za
   mały. Line search musi dramatycznie zwiększyć `alpha` (>> 1) albo
   algorytm proponuje kierunek dominowany przez inną zmienną.

3. Następna iteracja: znów inna zmienna dominuje, `theta` się zmienia,
   i cykl się powtarza.

Efekt: `alpha` skacze chaotycznie, bo w każdej iteracji model jest dobrze
skalibrowany dla jednej grupy zmiennych, ale źle dla pozostałych. Line search
musi nieustannie kompensować ten brak równowagi.

Z dokładną diagonalą ten problem znika: `theta * diag(h)` jest dobrze
skalowane dla WSZYSTKICH zmiennych jednocześnie. Model dobrze przewiduje
funkcję we wszystkich kierunkach, więc `alpha ≈ 1` w każdej iteracji.

## Wykresy

Zapisane w `plots/`:
- `lbfgsb_initial_hessian_benchmark.png` --- 4-panelowy wykres zbieżności
- `lbfgsb_initial_hessian_bar.png` --- wykres słupkowy porównania ewaluacji
