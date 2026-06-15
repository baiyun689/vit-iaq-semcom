"""冒烟测试：确认包结构可正常导入。"""

def test_import_package():
    import vit_iaq_semcom
    assert vit_iaq_semcom.__version__
