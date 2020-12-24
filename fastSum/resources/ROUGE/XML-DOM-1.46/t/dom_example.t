BEGIN {print "1..5\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
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
<DATA>
<CKL>
<CKLID>P001</CKLID>
<SEGMENT> </SEGMENT>
<COUNTRY>USA</COUNTRY>
<LOCALCONTACT>
<ADDRESS>HNLLHIWP</ADDRESS>
</LOCALCONTACT>
</CKL>
<CKL>
<CKLID>0002</CKLID>
<SEGMENT> </SEGMENT>
<COUNTRY>USA</COUNTRY>
<LOCALCONTACT>
<ADDRESS>45 HOLOMOA STREET</ADDRESS>
</LOCALCONTACT>
</CKL>
</DATA>
END

my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);
assert_ok (not $@);

my $error = 0;
my $ckls = $doc->getElementsByTagName ("CKL");
assert_ok ($ckls->getLength == 2);
for my $ckl (@$ckls)
{
    my $cklids = $ckl->getElementsByTagName ("CKLID");
    my $cklid = $cklids->[0]->getFirstChild->getData;
    $error++ if ($cklid ne "P001" && $cklid ne "0002");

    my $countries = $ckl->getElementsByTagName ("COUNTRY");
    my $country = $countries->[0]->getFirstChild->getData;
    $error++ if ($country ne "USA");
}
assert_ok ($error == 0);

# Use getElementsByTagName in list context
my @ckls = $doc->getElementsByTagName ("CKL");
assert_ok (@ckls == 2);
