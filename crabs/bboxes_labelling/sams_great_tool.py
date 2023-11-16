import typer


def doesnt_need_to_be_called_main(
    input_filename: str,
    output_filename: str,
    something_else: str,
    a_bool: bool,
) -> None:
    print(f"{input_filename} → {output_filename}")
    if a_bool:
        print(something_else)
    return


def main():
    typer.run(doesnt_need_to_be_called_main)


if __name__ == "__main__":
    main()
