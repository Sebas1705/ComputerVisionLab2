from Classes.Ex3.PanelsLoader import PanelsLoader

def test() -> None:
    
    #Charge all test images:
    p = PanelsLoader("test_ocr_panels_origen\\","test_ocr_panels_new\\",True)
    images = p.get_proccess_panels()
