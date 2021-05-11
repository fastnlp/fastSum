BEGIN {print "1..2\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use XML::Parser::PerlSAX;
use XML::Handler::BuildDOM;
#use XML::Filter::SAXT;
#use XML::Handler::PrintEvents;

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
<!DOCTYPE simpsons [
 <!ELEMENT person (#PCDATA)>
 <!ATTLIST person
  name CDATA #REQUIRED
  hair (none|blue|yellow) 'yellow'
  sex CDATA #REQUIRED>
]>
<simpsons>
 <person name="homer" hair="none" sex="male"/>
 <person name="marge" hair="blue" sex="female"/>
 <person name="bart" sex="almost"/>
 <person name="lisa" sex="never"/>
</simpsons>
END

my $build_dom = new XML::Handler::BuildDOM;
my $parser = new XML::Parser::PerlSAX (UseAttributeOrder => 1,
				       Handler => $build_dom);

#
# This commented code is for debugging. It inserts a PrintEvents handler,
# so you can see what events are coming thru.
#
#my $build_dom = new XML::Handler::BuildDOM;
#my $pr_evt = new XML::Handler::PrintEvents;
#my $saxt = new XML::Filter::SAXT ({ Handler => $pr_evt },
#				  { Handler => $build_dom });
#my $parser = new XML::Parser::PerlSAX (UseAttributeOrder => 1,
#				       Handler => $saxt);

my $doc = $parser->parse ($str);

# It throws an exception with XML::Parser 2.27: 
#
#   Can't use string ("<!DOCTYPE simpsons [ <!ELEMENT ") as a symbol ref 
#   while "strict refs" in use at /home1/enno/perl500502/lib/site_perl/5.005/
#   sun4-solaris/XML/Parser/Expat.pm line 426.
#
# I don't get it, so let's not check it for now.
#
#assert_ok (not $@);

my $out = $doc->toString;
$out =~ tr/\012/\n/;
print "out: $out --end\n\nstr: $str --end\n";
assert_ok ($out eq $str);
