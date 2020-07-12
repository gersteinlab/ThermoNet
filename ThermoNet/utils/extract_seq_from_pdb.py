#!/usr/bin/env python3

# import required modules
from Bio import PDB, SeqIO, SeqRecord, pairwise2
from argparse import ArgumentParser
from os.path import basename


def main():
    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('-p', '--pdb', dest='pdb',
        help='input PDB file')
    parser.add_argument('-c', '--chains', dest='chains',
        help='chains to extract sequence for')
    parser.add_argument('-o', '--output', dest='output',
        help='output fasta file')
    parser.add_argument('-r', '--resseq', dest='resseq',
        help='residue seq number in the PDB file')
    parser.add_argument('-i', '--interactive', dest='interactive',
        action='store_true', help='select sequenceinteractively')
    args = parser.parse_args()

    print( "input file:          " + args.pdb)
    print( "output file:         " + args.output )
    print( "chains:              " + args.chains )

    # extract sequences from SEQRES records
    pdb_id = basename(args.pdb).split('.')[0]
    with open(args.pdb, 'rt') as f:
        seqres_sequences = list(SeqIO.parse(f, 'pdb-seqres'))

    # extract sequences from residues with resolved coordinates
    structure = PDB.PDBParser().get_structure(id=pdb_id, file=args.pdb)
    model = structure[0]
    peptide_builder = PDB.Polypeptide.PPBuilder()
    sequences = []
    resseq_ids = []
    for c in args.chains:
        residues = [r for r in model[c].get_residues() if PDB.is_aa(r)]
        resseq_ids.append(r.get_id()[1] for r in residues)
        chain = peptide_builder.build_peptides(model[c])
        coord_sequence = chain[0].get_sequence()
        print('Sequence for chain ' + c + ' extracted from coordinates:')
        print(coord_sequence, '\n')
        # if the SEQRES records are missing from the PDB file, use sequence from coordinates
        if len(seqres_sequences) == 0:
            print('SEQRES records in the given PDB file are missing! Using sequence '
                  'extracted from coordinates.')
            sequence = coord_sequence
        else:
            for record in seqres_sequences:
                if c == record.id:
                    print('Sequence for chain ' + c + ' in SEQRES:')
                    print(record.seq, '\n')
            # pairwise alignment between the two sequences
            print('Here is an alignment of the two sequences:')
            alignment = pairwise2.align.globalms(coord_sequence, record.seq, 1, -0.5, -10, 0)
            print(pairwise2.format_alignment(*alignment[0]))


            if args.interactive:
                # ask for which sequence to choose
                s = input('Which sequence are you interested? 1 for sequence from coordinates '
                          '2 for sequence from SEQRES: ')
                if int(s) == 1:
                    sequence = coord_sequence
                else:
                    sequence = record.seq
            else:
                sequence = record.seq

        # append sequence for the current chain
        sequences.append(
                SeqRecord.SeqRecord(
                        seq=sequence,
                        id=pdb_id.upper() + ':' + c,
                        description=''))

    # write sequences to a fasta file
    with open(args.output, 'wt') as f:
        SeqIO.write(sequences, f, 'fasta')

    # write resseq ids
    if args.resseq is not None:
        with open(args.resseq, 'wt') as f:
            for i, resseq in enumerate(resseq_ids):
                f.write('> ' + args.chains[i] + '\n')
                f.write(','.join(str(j) for j in resseq))


if __name__ == '__main__':
    main()
