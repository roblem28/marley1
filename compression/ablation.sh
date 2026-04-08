#!/bin/bash
PIPELINE=~/marley1/compression/pipeline_pdf.py
SPECS=~/marley1/specs
RESULTS=~/marley1/compression/ablation_results.txt

# Disable relevance filter
sed -i 's|SKIP_RELEVANCE = False|SKIP_RELEVANCE = True|' $PIPELINE

echo "=== ABLATION: RELEVANCE FILTER OFF ===" > $RESULTS

for spec in electrical structural microgrid c5isr_facilities unaccompanied_housing; do
    query=""
    case $spec in
        electrical) query="What are the electrical design requirements for DoD facilities?" ;;
        structural) query="What are the structural design requirements for DoD buildings?" ;;
        microgrid) query="What are the design requirements for installation microgrids?" ;;
        c5isr_facilities) query="What are the requirements for C5ISR facility design?" ;;
        unaccompanied_housing) query="What are the design standards for military unaccompanied housing?" ;;
    esac
    sed -i "s|PDF_PATH = .*|PDF_PATH = os.path.expanduser(\"$SPECS/ufc_${spec}.pdf\")|" $PIPELINE
    sed -i "s|QUERY = .*|QUERY = \"$query\"|" $PIPELINE
    echo "--- $spec ---" >> $RESULTS
    python3 $PIPELINE 2>&1 | grep -E "Total tokens|Total compression|FAT MAN RESPONSE" -A 10 >> $RESULTS
done

# Re-enable relevance filter
sed -i 's|SKIP_RELEVANCE = True|SKIP_RELEVANCE = False|' $PIPELINE
echo "Done. Results in $RESULTS"
