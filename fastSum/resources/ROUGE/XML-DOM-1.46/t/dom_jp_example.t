BEGIN {print "1..5\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use utf8;
$loaded = 1;
print "ok 1\n";

my $test = 1;
sub assert_ok
{
    my $ok = shift;
    print "not " unless $ok;
    ++$test;
    print "ok $test\n";
    $ok;
}

#Test 2

my $str = <<END;
<データ>
<商品>
<商品番号>P001</商品番号>
<ジャンル> </ジャンル>
<生産国>米国</生産国>
<国内連絡先>
<住所>隣のアパート</住所>
</国内連絡先>
</商品>
<商品>
<商品番号>0002</商品番号>
<ジャンル> </ジャンル>
<生産国>米国</生産国>
<国内連絡先>
<住所>横須賀市 光の丘</住所>
</国内連絡先>
</商品>
</データ>
END

my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);
assert_ok (not $@);

my $error = 0;
my $ckls = $doc->getElementsByTagName ("商品");
assert_ok ($ckls->getLength == 2);
for my $ckl (@$ckls)
{
    my $cklids = $ckl->getElementsByTagName ("商品番号");
    my $cklid = $cklids->[0]->getFirstChild->getData;
    $error++ if ($cklid ne "P001" && $cklid ne "0002");

    my $countries = $ckl->getElementsByTagName ("生産国");
    my $country = $countries->[0]->getFirstChild->getData;
    $error++ if ($country ne "米国");
}
assert_ok ($error == 0);

# Use getElementsByTagName in list context
my @ckls = $doc->getElementsByTagName ("商品");
assert_ok (@ckls == 2);
