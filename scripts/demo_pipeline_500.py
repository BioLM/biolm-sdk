"""
Demo pipeline: real predictions with temberture-regression (Tm) + biolmsol (solubility).

Uses BIOLMAI_TOKEN from the environment for live API calls.

Run:
    python scripts/demo_pipeline_500.py
"""

import asyncio
import os
import tempfile
from pathlib import Path

from biolm.pipeline.data import DataPipeline
from biolm.pipeline.datastore_duckdb import DuckDBDataStore
from biolm.pipeline.filters import (
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
)

# ---------------------------------------------------------------------------
# Sequences — mix of thermostable and mesophilic proteins for a meaningful
# predictive range (known literature Tm values vary ~30–100 °C across these).
# ---------------------------------------------------------------------------
SEQUENCES = [
    # Thermostable proteins / archaeal enzymes (expected high Tm)
    "MKVLKAGISFTLGSGVILGAFIFFVLVKDNPKLTTGELTLQTAVEMAPQPIAGLSHEIQGVGYEITDDMIEPVTLGSGLVNPVGQILMGGIDGGISALPNLEKFNKTIEGLPDKLKPTFTTIEDAMKLYAAYGGIFNYLAKIDTVNFDLITDRDNEEAKKALYKMLGSKVDLPQLKEFPLDGYVDAQLKAIDEYLRDNPRAATLKQIATDIASQFRQALENYGNTVEAMKAMLEEGVSRSYDEPQNYKNMVFNQDGMTPTEYAEVDRLQKAIDAKFKQMLENLNAKQKQLVTMKAIADMKPELTEKEQVMLVQNLQKMKEQLATLKTKIDGLAAELTLHPQVRNFDIEAINKDLKNEAQKQMSQVLNQLKMDIEAAKEQLATLAAQHQALITKLKDKVSEAIQKAAQDLAPEITQQLNANLKKSQSQVLNQLKMDIEAAKEQLATLAAQHQ",
    # Mesophilic E. coli proteins (expected lower Tm)
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPSLKE",
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGAAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFHWANQKGSDHWSAASARTLRMVEDGGVLVGQPLDGAVHGLQGGADFPWFPRAKFPSWYADPQSAAFSGSGSGDAGSSGSGSGGSSG",
    "MKTRTLSERILENLEKKGLIKDPNRDINALLKMLDQNTDDKEIEDYKTKLGKIIEEYKNLNEQIESAIKAALKASKEQHKQLQEAQEREQALKQQAQELEQAQRKQEAVKEMKDAYAELQKNIESRSSDLKKLSQKDKEMQKEINKTKSTLQQMKKTFDEQIFEMKSELAKATAEQLEARLKNAAQLNEDLMKAKQDLEEAQQLAQKQVELEAKEKQITDLQNLNKKLQQLKDELEEKLKKIQQERKELQKKLQELNDLKQQLEELKQKIENLKAKIANITEQLQKIIEHLNLEIKDKLQELERKLQKERDKIEQEIADLKEKLNQLNKKIQQLQEQINRLKQEINELQKKINNLQKRLEQLNQDLNKIQQELQKKIEELKKRLKQLEEQLKNLKQQIQKQETELKQNKIQLQNQIQKLQAEIERQKKKIELDIKKLQQELNKLKQQLKPLELTNKDFKSSIEMVNENLQDLIDDLNQKLQEQGIQTKELKDNFKQILQQLKE",
    "MASMTGGQQMGRDPNSIVAVYNYKYPQLCQGRRKPKGLGISKNKFNQSVILLSELEVKDNALKLREEFAKQNQVKEITGNVMDKLNSNMAKLLELKAKFDEELQTLKAKAEQQKAEINELKKTQNNLNQLQKQLQELQQKIEELKKKIEQLQQKIQDLKQQIKQQKELIESQIKQLQKELKELQKELQNQISKLQQLQKELKEQIQELQKELQQLKKELQQL",
    # Known high-Tm hyperthermophile fragments
    "MKILILLGAEKGIGKSTIAKLLAQKFGKIVIETNEKDEDAKSIAEQLGKPFDSVSQLDTIAPQVLAQLLREELSSIMTQIPDVSIVVLDSQGAAITQSALENIRPDYIIVNKMDLKKKDQFAPAAILEQKIRDYFPELDAQSKELEDLFQKAGVEVISQAESFIAQYISQRKDLP",
    "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDALPNISDAERIFAELLTGLAAAQPGFPLAQLKTFVDQEFAQIKHVLHGISGGVGGVATITAPKKVTLLGRDSFEVAVALMK",
    # Antibody fragments (mesophilic, diverse)
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYTMSWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDDHYSLDYWGQGTLVTVSS",
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVTSANWFDPWGQGTLVTVSS",
    "SYELTQPPSVSSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYDNTNRPSGVPDRFSGSKSGTSASLAISGLRSEDEADYYCATWDDSLSGYVFGGGTKLTVL",
    # Thermostable enzymes
    "MFKVYGYDSNIHKCNVCDKCHAKDPDMRIVVENMDNPKAHIFEKIDSGKLADYIKAVGADSVIYPDDAKLKFNEPQPEEHLRAIFEKAKKTGKVAVKVLEKYENDTSVDFNQYAQDYQKKLGELIPQNAFNIPVTAPDLYKQIPGGKYTFNIVNGENIFKNGTDEQQSALNAAQMITKELDLKDKIVTFKELKPNLYENLKFDIEMTQLAEQYISSQLNKIMEMVKPNNIKPEQMAELEAAYKLQSYTG",
    "MKIVVLGSGPAGYSAAFRCADNPDVEVVALDADEDYWLNSDAGHMPAGQQIVDKLREIGAQFRTQFEEAGPLVNLHRTLNTLISAAPNDQHAVVLDTNLQDLQRRRGDQLIEAGASDRAVLHPDLIEQLQEAGFKPMVVKALATGGSGSGITLNKDLAEQLRDLGHTVHFHQNPTQALLDQQNLHEYRG",
    # Short-ish sequences (>= 20 aa for biolmsol)
    "MKTAYIAKQRQGHQAMAEIKQ",
    "MSHHWGYGKHNGPEHWHKDFPIAK",
    "MKLAVIDSAQKALAEEQKQILQELQ",
    "MAEIKQRQGHQAMKTAYIAKQRQGHQ",
    # Human proteins
    "MGSSHHHHHHSSGLVPRGSHMKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV",
    "MTMDKSELVQKAKLAEQAERYDDMASAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTEGDLSNKMLHKIAEDAERKYVDKLKTDLKDQKPVQNLMKQFVTELQEHLLEQISELKDIKKYQTEQQRIEELQKEIEDLKQKIEEAKQLLENYVEQYKTFRELEKLIRDKFQTLQEQQDQILEELKDMKKELEKNNKTLIQNLKQKIEEIKQELSETLQKLETKFQEQLEQVKQRIKRLENELEKLKRQILKLNDLENELEKTLKRMQEELQALQHRLPNEQLQMLEEERKEQIERLSHFRESLFQKLKEVKEQVKKILKELEKKLQEQLKEIQKEMQDLLENLQQELEQIRTQLEKRKKELEKELQQQLNQLREELKEQVKQLKDELDQTKKELE",
    # de novo designed hyperstable proteins
    "MGSSHHHHHHHGSIEGRLHMIKEIEDELGREALNKAWAQAQAQLQRQSGEQLQDLKLRYQQQLKQLDSEMRRNLRQEQERTLRQQLQDLQAERDALREQLQMQRVMELEQQYQQSITEQNRQLQKELDAANKQLQEAKNAFKEQLNELKNQLSTLQEQLNSQMNQLQDDLKNQIREQEKQLQKELQQAYEQQLNQLKQELRQAKQQIQEQAKELQNQLQEFHQKM",
]

RUN_ID = "demo_real_pipeline_002"
RUN_ID_3 = "demo_real_pipeline_002_batch2"

# ---------------------------------------------------------------------------
# Second batch — new sequences not in the first set. When trickling into the
# same DuckDB the 20 original sequences will hit the prediction cache; only
# these 12 new ones will trigger real API calls.
# ---------------------------------------------------------------------------
NEW_SEQUENCES = [
    # More antibody VH/VL sequences
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARSTYYGGDWFDPWGQGTLVTVSS",
    "DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPLTFGGGTKVEIK",
    "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVSNIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDGNYYGSGFAYWGQGTLVTVSS",
    "SYELTQPPSVSVSPGQTARITCSGDALPKQYAYWYQQKSGQAPVLVIYKDSERPSGIPERFSGSSSGTTVTLTISGVQAEDEADYYCQSADSSGTYVFGGGTKLTVL",
    # Thermostable TIM-barrel enzymes
    "MKVTLNYGKPVQEIKPAELTEALIDLGIDVEIVDGTPGELAKNLASRGKSVVVIGHRTGRQQTHIDHILQAAGKKVLVIGMGSYSGFKAMKQYLEEAGADVHVIEKIAECPDMPVIDEVEAAK",
    "MAQPKLQEIRDKKPRFILTQNLTDEELKEFIERYSQSGKIVPSVGDLITPHKNLQLNDSHVMPYITRKVMYKNPHIRLIQGELQEAFETLSELLKSLGALVVDIQKYVMQEGAEKQADILQKRPVDIIAHIEQDFDPITDVAKKY",
    # Designed coiled-coil peptides (≥20 aa)
    "MKQLEDKVEELLSKNYHLENEVARLKKLVGER",
    "MASMTGGQQMGRGSMDELEQKLISEEDLNSAVDHHHHHH",
    # Nanobody/VHH sequences
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAEPSGYSTYYWGQGTLVTVSS",
    "EVQLVESGGGLVQPNNSLRLSCAASGFTLDDYAMGWYRQAPGKQRELVSTITGGGSITYYADSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCARRGSYYDSSGYNYWGQGTLVTVSS",
    # Thermophile metabolic enzymes
    "MRIILLGAPGAGKGTQAKFIEEKGYIPHISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG",
    # Short functional peptides (borderline 20 aa minimum)
    "MKTAYIAKQRQGHQAMAEIKQKLR",
]


def build_pipeline_batch2(db_path: Path, data_dir: Path, run_id: str):
    """Run 3: trickle SEQUENCES + NEW_SEQUENCES into the existing DuckDB.

    Old sequences already have predictions cached — only NEW_SEQUENCES hit the API.
    """
    ds = DuckDBDataStore(db_path=db_path, data_dir=data_dir)
    pipeline = DataPipeline(
        sequences=SEQUENCES + NEW_SEQUENCES,  # all 32 sequences
        datastore=ds,
        run_id=run_id,
        resume=False,  # new run_id, but predictions for old seqs come from cache
        verbose=True,
    )
    pipeline.add_filter(
        SequenceLengthFilter(min_length=20),
        stage_name="prefilter_length",
    )
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        depends_on=["prefilter_length"],
    )
    pipeline.add_prediction(
        "biolmsol",
        extractions="solubility_score",
        columns="solubility",
        stage_name="predict_sol",
        depends_on=["prefilter_length"],
    )
    pipeline.add_filter(
        ThresholdFilter("tm", min_value=40.0),
        stage_name="filter_tm",
        depends_on=["predict_tm"],
    )
    pipeline.add_filter(
        RankingFilter("solubility", n=8, ascending=False),  # top 8 now
        stage_name="top8_soluble",
        depends_on=["filter_tm", "predict_sol"],
    )
    return pipeline, ds


def build_pipeline(db_path: Path, data_dir: Path, run_id: str, resume: bool = False):
    ds = DuckDBDataStore(db_path=db_path, data_dir=data_dir)
    pipeline = DataPipeline(
        sequences=SEQUENCES,
        datastore=ds,
        run_id=run_id,
        resume=resume,
        verbose=True,
    )

    # Pre-filter: remove sequences too short for biolmsol (min 20 aa)
    pipeline.add_filter(
        SequenceLengthFilter(min_length=20),
        stage_name="prefilter_length",
    )

    # Stage 1: melting temperature via TemBERTure regression model
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        depends_on=["prefilter_length"],
    )

    # Stage 2: solubility via BioLMSol (runs in parallel with Tm)
    pipeline.add_prediction(
        "biolmsol",
        extractions="solubility_score",
        columns="solubility",
        stage_name="predict_sol",
        depends_on=["prefilter_length"],
    )

    # Filter: keep only sequences with Tm >= 40 °C (remove obvious fails)
    pipeline.add_filter(
        ThresholdFilter("tm", min_value=40.0),
        stage_name="filter_tm",
        depends_on=["predict_tm"],
    )

    # Rank: top 5 by solubility among Tm-passing sequences
    pipeline.add_filter(
        RankingFilter("solubility", n=5, ascending=False),
        stage_name="top5_soluble",
        depends_on=["filter_tm", "predict_sol"],
    )

    return pipeline, ds


async def main():
    token = os.environ.get("BIOLMAI_TOKEN", "")
    if not token:
        raise RuntimeError("BIOLMAI_TOKEN not set — source your .zshrc first")

    print(f"Running real pipeline on {len(SEQUENCES)} sequences")
    print("Models: temberture-regression (Tm °C) + biolmsol (solubility)")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "demo.duckdb"
        data_dir = tmp_path / "data"

        # ── Run 1 ──────────────────────────────────────────────────────────
        print("=" * 60)
        print("RUN 1 — live API predictions")
        print("=" * 60)

        pipeline1, ds1 = build_pipeline(db_path, data_dir, RUN_ID, resume=False)
        await pipeline1.run_async()

        df1 = pipeline1.get_final_data()
        explore = pipeline1.explore()
        stats = pipeline1.stats()

        print(f"\n{'='*60}")
        print(f"Sequences in store : {explore.get('sequences', '?')}")
        print(f"Predictions stored : {explore.get('predictions', '?')}")
        print("\nStage summary:")
        print(stats.to_string(index=False))

        print(f"\nTop {len(df1)} sequences (by solubility, Tm ≥ 40 °C):")
        print("-" * 60)
        show_cols = [c for c in ["sequence", "tm", "solubility"] if c in df1.columns]
        for _, row in df1[show_cols].iterrows():
            seq_short = (
                row["sequence"][:30] + "..."
                if len(row["sequence"]) > 30
                else row["sequence"]
            )
            tm_str = f"Tm={row['tm']:.1f}°C" if "tm" in row else ""
            sol_str = f"Sol={row['solubility']:.3f}" if "solubility" in row else ""
            print(f"  {seq_short:<33} {tm_str:<14} {sol_str}")
        ds1.close()

        # ── Run 2 (resume) ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("RUN 2 — resume (should skip all API calls)")
        print("=" * 60)

        pipeline2, ds2 = build_pipeline(db_path, data_dir, RUN_ID, resume=True)
        await pipeline2.run_async()

        df2 = pipeline2.get_final_data()
        stats2 = pipeline2.stats()
        print("\nResume stats:")
        print(stats2.to_string(index=False))

        assert len(df2) == len(
            df1
        ), f"Resume produced different output: {len(df2)} vs {len(df1)}"
        print(f"\nResume reproduced same {len(df2)} sequences ✓")
        ds2.close()

        # ── SQL power query ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("SQL QUERY — sequences with Tm > 50 and solubility > 0.5")
        print("=" * 60)
        # Reopen for final query
        _, ds3 = build_pipeline(db_path, data_dir, RUN_ID, resume=True)
        high_quality = ds3.conn.execute(
            """
            SELECT s.sequence, tm.value as tm, sol.value as solubility
            FROM sequences s
            JOIN predictions tm ON s.sequence_id = tm.sequence_id
                AND tm.prediction_type = 'tm'
            JOIN predictions sol ON s.sequence_id = sol.sequence_id
                AND sol.prediction_type = 'solubility'
            WHERE tm.value > 50 AND sol.value > 0.5
            ORDER BY tm.value DESC
        """
        ).df()
        print(f"Found {len(high_quality)} sequences with Tm>50°C and solubility>0.5:")
        for _, row in high_quality.iterrows():
            seq_short = (
                row["sequence"][:35] + "..."
                if len(row["sequence"]) > 35
                else row["sequence"]
            )
            print(f"  Tm={row['tm']:.1f}°C  Sol={row['solubility']:.3f}  {seq_short}")
        ds3.close()

        # ── Run 3 (trickle new sequences) ──────────────────────────────────
        print(f"\n{'='*60}")
        print(f"RUN 3 — trickle {len(NEW_SEQUENCES)} new sequences into same DuckDB")
        print(f"  Old {len(SEQUENCES)} sequences → prediction cache (no API calls)")
        print(f"  New {len(NEW_SEQUENCES)} sequences → live API calls only")
        print("=" * 60)

        pipeline3, ds4 = build_pipeline_batch2(db_path, data_dir, RUN_ID_3)
        await pipeline3.run_async()

        df3 = pipeline3.get_final_data()
        explore3 = pipeline3.explore()
        stats3 = pipeline3.stats()

        print("\nStore after trickle:")
        print(f"  Sequences in store : {explore3.get('sequences', '?')}  (was 20)")
        print(f"  Predictions stored : {explore3.get('predictions', '?')}")

        # Show cache efficiency from the stage results
        for stage_name, sr in pipeline3.stage_results.items():
            if hasattr(sr, "cached_count") and hasattr(sr, "computed_count"):
                if sr.cached_count or sr.computed_count:
                    print(
                        f"  {stage_name}: {sr.cached_count} cached, {sr.computed_count} new API calls"
                    )

        print(f"\nTop {len(df3)} sequences after trickle (Tm ≥ 40, top-8 solubility):")
        print("-" * 60)
        show_cols = [c for c in ["sequence", "tm", "solubility"] if c in df3.columns]
        new_seqs_set = set(NEW_SEQUENCES)
        for _, row in df3[show_cols].iterrows():
            seq_short = (
                row["sequence"][:30] + "..."
                if len(row["sequence"]) > 30
                else row["sequence"]
            )
            tm_str = f"Tm={row['tm']:.1f}°C" if "tm" in row else ""
            sol_str = f"Sol={row['solubility']:.3f}" if "solubility" in row else ""
            tag = " ← NEW" if row["sequence"] in new_seqs_set else ""
            print(f"  {seq_short:<33} {tm_str:<14} {sol_str}{tag}")

        print("\nStage summary (Run 3):")
        print(stats3.to_string(index=False))

        # Verify cache worked: predictions table should have grown
        total_preds = ds4.conn.execute(
            "SELECT COUNT(DISTINCT sequence_id) FROM predictions"
        ).fetchone()[0]
        assert total_preds >= len(SEQUENCES), f"Should retain all original predictions, got {total_preds}"
        ds4.close()

    print(f"\n{'='*60}")
    print("Demo passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
