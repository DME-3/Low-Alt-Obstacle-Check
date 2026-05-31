def test_osn_data_update_import_is_side_effect_free():
    import OSN_data_update

    assert callable(OSN_data_update.main)

