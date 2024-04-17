
- Para executar o teste de memória, é necessário ter o valgrind instalado. Para instalar no Ubuntu, execute o comando:
    ```bash
    sudo apt-get install valgrind
    ```

- Para executar o teste de memória, execute o comando, dentro da pasta `tests`:
    ```bash
    valgrind --tool=memcheck --leak-check=full --suppressions=valgrind-python.supp --log-file=minimal.valgrind.log python utils/try_one.py 
    ```
- O resultado do teste será salvo no arquivo `minimal.valgrind.log`.
