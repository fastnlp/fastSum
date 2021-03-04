from summarizationLoader import  ArxivLoader

if __name__ == '__main__':
    ArxivLoader().download()
    data = ArxivLoader().load()
    print(data)