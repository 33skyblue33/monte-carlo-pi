# Szacowanie liczby PI metodą Monte Carlo

Projekt ten dostarcza trzy różne implementacje metody Monte Carlo do szacowania wartości liczby PI:

1.  **Sekwencyjna na CPU:** Implementacja jednowątkowa, działająca na procesorze CPU.
2.  **Równoległa na CPU:** Implementacja wielowątkowa, która wykorzystuje wiele rdzeni procesora, aby przyspieszyć obliczenia.
3.  **Równoległa na GPU:** Masywnie równoległa implementacja, która działa na karcie graficznej (GPU) przy użyciu biblioteki **PyTorch**.

## Jak to działa?

Metoda Monte Carlo do szacowania PI to algorytm probabilistyczny oparty na symulacji. Główna idea polega na wpisaniu okręgu w kwadrat, a następnie wylosowaniu ogromnej liczby punktów wewnątrz tego kwadratu. Stosunek liczby punktów, które znalazły się wewnątrz okręgu, do całkowitej liczby wylosowanych punktów jest w przybliżeniu równy stosunkowi pola powierzchni okręgu do pola powierzchni kwadratu.

Ponieważ pole okręgu wynosi πr², a pole kwadratu (2r)² = 4r², stosunek pól wynosi πr² / 4r² = π/4. Dzięki temu możemy oszacować wartość PI jako:

**PI ≈ 4 \* (liczba punktów wewnątrz okręgu) / (całkowita liczba punktów)**

## Instalacja

Zdecydowanie zalecane jest użycie menedżera pakietów `conda`, ponieważ automatycznie zarządza on złożonymi zależnościami CUDA wymaganymi przez PyTorch.

### Opcja 1: Środowisko dla CPU (bez wsparcia GPU)

Jeśli chcesz uruchomić tylko wersje na CPU, możesz użyć prostszego środowiska.

1.  **Utwórz środowisko wirtualne:**
    ```bash
    python -m venv venv
    ```
2.  **Aktywuj środowisko:**
    *   Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
3.  **Zainstaluj zależności:**
    ```bash
    pip install tqdm
    ```

### Opcja 2: Środowisko dla GPU (z PyTorch - zalecane)

To środowisko pozwoli na uruchomienie wszystkich trzech wersji algorytmu.

1.  **Utwórz nowe środowisko `conda`:**
    Poniższa komenda stworzy nowe środowisko o nazwie `torch_env` z odpowiednią wersją Pythona, PyTorch i narzędzi CUDA.
    ```bash
    conda create --name torch_env python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
2.  **Aktywuj środowisko `conda`:**
    ```bash
    conda activate torch_env
    ```

## Uruchamianie algorytmów

Główny skrypt `main.py` służy do uruchamiania wszystkich algorytmów. Wybierz algorytm, podając jego nazwę jako pierwszy argument.

### CPU sekwencyjny

```bash
python main.py cpu --num_samples <liczba_próbek>
```

### CPU równoległy

Możesz opcjonalnie podać liczbę procesów roboczych (`--num_workers`). Domyślnie używana jest liczba rdzeni procesora.

```bash
python main.py cpu-parallel --num_samples <liczba_próbek> --num_workers <liczba_procesów>
```

### GPU równoległy (z PyTorch)

Możesz wybrać konkretne urządzenie GPU za pomocą flagi `--device`.

```bash
python main.py gpu --num_samples <liczba_próbek> --device <id_gpu>
```

**Uwaga:** Argument `--threads_per_block` jest ignorowany w wersji z PyTorch, ale został zachowany dla spójności interfejsu.

### Listowanie dostępnych GPU

Aby wyświetlić listę dostępnych kart graficznych wykrytych przez PyTorch, użyj flagi `--list_gpus`.

```bash
python main.py --list_gpus
```