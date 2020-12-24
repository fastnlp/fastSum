BEGIN {print "1..3\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
$loaded = 1;
print "ok 1\n";

#Test 2

my $str = <<END;
<?xml version="1.0" standalone="yes"?>
<!DOCTYPE doc [
 <!ENTITY huh "Uh huh huh huh mmh huh">
 <!ELEMENT doc (beavis|butthead)*>
 <!ELEMENT beavis (#PCDATA)>
 <!ELEMENT butthead (#PCDATA)>
]>
<doc>
 <beavis>
Hey Butthead!
 </beavis>
 <butthead>
Yes, Beavis.
 </butthead>  
 <beavis>
You farted. &huh;
 </beavis>
 <butthead>
&huh; Yeah &huh;
 </butthead>  
</doc>
END

my $parser = new XML::DOM::Parser (NoExpand => 1);
my $doc = $parser->parse ($str);
my $out = $doc->toString;
$out =~ tr/\012/\n/;

if ($out ne $str)
{
    print "not ";
}
print "ok 2\n";

$str = $doc->getElementsByTagName("butthead")->item(0)->toString;
$str =~ tr/\012/\n/;

if ($str ne "<butthead>\nYes, Beavis.\n </butthead>")
{
    print "not ";
}
print "ok 3\n";
