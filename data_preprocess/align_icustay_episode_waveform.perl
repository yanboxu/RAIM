#!/usr/local/bin/perl

use Date::Parse;

$waveform_record_file = '../MIMIC_data/waveform_matched_subset/RECORDS-waveforms';
$icustay_file = '../MIMIC_data/clinical_data/from_benchmarks/all_stays.csv';
$test_split_dir = '../MIMIC_data/clinical_data/from_benchmarks/test/';
$train_split_dir = '../MIMIC_data/clinical_data/from_benchmarks/train/';
$out_file = '../MIMIC_data/intermediate_data/ICUSTAYS.waveform_matched.csv';

%hashSPLIT = ();
%numICUSTAY = ();
%hashICUSTAY = ();

print "Reading split list...\n";
open( $fh, "-|", "find", $test_split_dir, "-type", "d" );
$line = <$fh>;
while ($line = <$fh>) {
        chomp $line;
        $line =~ s/$test_split_dir//g;
	$hashSPLIT{$line} = 'test';
}
close $fh;

open( $fh, "-|", "find", $train_split_dir, "-type", "d" );
$line = <$fh>;
while ($line = <$fh>) {
        chomp $line;
        $line =~ s/$train_split_dir//g;
        $hashSPLIT{$line} = 'train';
}
close $fh;

print "Reading ICUSTAY...\n";
open($ICUSTAY, "<", $icustay_file)
        or die "Can't open $icustay_file: $!";

$head = <$ICUSTAY>;
chomp $head;
while($line = <$ICUSTAY>){
	chomp $line;
	@tokens = split(/,/, $line);
	$subject_id = sprintf("p%06d", $tokens[0]);
	$intime = $tokens[5];
	$outtime = $tokens[6];
	
	if(!exists $numICUSTAY{$subject_id}){
		$numICUSTAY{$subject_id} = 1;
	}else{
		$numICUSTAY{$subject_id} = $numICUSTAY{$subject_id} + 1;
	}
	$episode_num = $numICUSTAY{$subject_id};

	$hashICUSTAY{$subject_id}->{$episode_num}->{ID} = $tokens[0];	
	$hashICUSTAY{$subject_id}->{$episode_num}->{LINE} = $line;	
	$hashICUSTAY{$subject_id}->{$episode_num}->{IN} = str2time($intime);	
	$hashICUSTAY{$subject_id}->{$episode_num}->{OUT} = str2time($outtime);	
	
}
close $ICUSTAY;

print "Writing to out file...\n";
open($OUT, ">", $out_file)
        or die "Can't open $out_file: $!";

print $OUT "SPLIT,EPISODE,RECORD_WAVEFORM_FILE,RECORD_WAVEFORM_START_TIME,$head\n";
open($LIST_RECORD, "<", $waveform_record_file)
        or die "Can't open $waveform_record_file: $!";
while($line = <$LIST_RECORD>){
	chomp $line;
	@tokens = split(/\//,$line);
	$subject_id = $tokens[1];
	@toks = split(/-/,$tokens[2]);
	$start_time = sprintf("%s-%s-%s %s:%s:00", $toks[1], $toks[2], $toks[3], $toks[4], $toks[5]);
	$int_time = str2time($start_time);

	if(exists $numICUSTAY{$subject_id}){
		$total_episode_num = $numICUSTAY{$subject_id};
		for(my $num = 1; $num <= $total_episode_num; $num++){
			$intime = $hashICUSTAY{$subject_id}->{$num}->{IN};
			$outtime = $hashICUSTAY{$subject_id}->{$num}->{OUT};
			if(($int_time >= $intime) & ($int_time <= $outtime)){
				$id = $hashICUSTAY{$subject_id}->{$num}->{ID};
				if(exists $hashSPLIT{$id}){
					$str = join ",",sprintf("\"%s\"",$hashSPLIT{$id}),sprintf("\"%s_episode%d_timeseries.csv\"",$id,$num),sprintf("\"%s\"",$line), $start_time,$hashICUSTAY{$subject_id}->{$num}->{LINE};
					print $OUT "$str\n";
				}
			}
		}
	} 
}
close $LIST_RECORD;
close $OUT;
