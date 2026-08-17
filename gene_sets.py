

# human cell cycle is based on the updated 2019 list built into Seurat V5
# mouse cell cycle is based on R Annotation Hub for mouse cell cycle annotation with the biomart Ensembl ID: AH116909
CELL_CYCLE_GENE_SETS = {
    "human": {
        "s_genes": (
            "MCM5", "PCNA", "TYMS", "FEN1", "MCM7", 
            "MCM4", "RRM1", "UNG", "GINS2", "MCM6", 
            "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU", 
            "HELLS", "RFC2", "POLR1B", "NASP", "RAD51AP1", 
            "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", 
            "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", 
            "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", 
            "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", 
            "CHAF1B", "MRPL36", "E2F8"),
        "g2m_genes": (
            "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", 
            "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", 
            "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", 
            "PIMREG", "SMC4", "CCNB2", "CKAP2L", "CKAP2", 
            "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", 
            "GTSE1", "KIF20B", "HJURP", "CDCA3", "JPT1", 
            "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", 
            "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", 
            "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", 
            "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", 
            "G2E3", "GAS2L3", "CBX5", "CENPA")
    },

    "mouse": {
        "s_genes": (
            "Cdc45", "Uhrf1", "Mcm2", "Slbp", "Mcm5", 
            "Pola1", "Gmnn", "Cdc6", "Rrm2", "Atad2", 
            "Dscc1", "Mcm4", "Chaf1b", "Rfc2", "Msh2", 
            "Fen1", "Hells", "Prim1", "Tyms", "Mcm6", 
            "Wdr76", "Rad51", "Pcna", "Ccne2", "Casp8ap2", 
            "Usp1", "Nasp", "Rpa2", "Ung", "Rad51ap1", 
            "Blm", "Pold3", "Rrm1", "Cenpu", "Gins2", 
            "Tipin", "Brip1", "Dtl", "Exo1", "Ubr7", 
            "Clspn", "E2f8", "Cdca7"),
        "g2m_genes": (
            "Ube2c", "Lbr", "Ctcf", "Cdc20", "Cbx5", 
            "Kif11", "Anp32e", "Birc5", "Cdk1", "Tmpo", 
            "Hmmr", "Jpt1", "Pimreg", "Aurkb", "Top2a", 
            "Gtse1", "Rangap1", "Cdca3", "Ndc80", "Kif20b", 
            "Cenpf", "Nek2", "Nuf2", "Nusap1", "Bub1", 
            "Tpx2", "Aurka", "Ect2", "Cks1b", "Kif2c", 
            "Cdca8", "Cenpa", "Mki67", "Ccnb2", "Kif23", 
            "Smc4", "G2e3", "Tubb4b", "Anln", "Tacc3", 
            "Dlgap5", "Ckap2", "Ncapd2", "Ttk", "Ckap5", 
            "Cdc25c", "Hjurp", "Cenpe", "Ckap2l", "Cdca2", 
            "Hmgb2", "Cks2", "Psrc1", "Gas2l3")
    }
}